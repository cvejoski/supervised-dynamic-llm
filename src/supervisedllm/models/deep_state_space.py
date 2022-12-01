import torch
from torch import nn
from supervisedllm.models.blocks import MLP
from supervisedllm.utils.reparametrizations import  gumbel_softmax, reparameterize_kumaraswamy, reparameterize_normal

KUMARASWAMY_BETA = "kumaraswamy-beta"
BERNOULLI = "bernoulli"
NORMAL = "normal"

EPSILON = 1e-10
EULER_GAMMA = 0.5772156649015329

def kullback_leibler(mean, logvar, reduction="mean"):
    """
    Kullback-Leibler divergence between Gaussian posterior distr.
    with parameters (mean, sigma) and a fixed Gaussian prior
    with mean = 0 and sigma = 1
    """

    kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())  # [B, D]
    skl = torch.sum(kl, dim=-1)
    if reduction == "mean":
        return torch.mean(skl)
    elif reduction == "sum":
        return torch.sum(skl)
    else:
        return skl

def kullback_leibler_two_gaussians(mean1, logvar1, mean2, logvar2, reduction="mean"):
    """
    Kullback-Leibler divergence between two Gaussians
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    kl = -0.5 * (1 - logvar2 + logvar1 - ((mean1 - mean2).pow(2) + var1) / var2)  # [B, D]
    skl = torch.sum(kl, dim=-1)
    if reduction == "mean":
        return torch.mean(skl)
    elif reduction == "sum":
        return torch.sum(skl)
    else:
        return skl

def beta_fn(a, b):
    beta_ab = torch.exp(torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    return beta_ab

def kl_kumaraswamy_beta(a, b, prior_alpha, prior_beta, reduction="mean"):
    # compute taylor expansion for E[log (1-v)] term
    # hard-code so we don't have to use Scan()
    kl = 1.0 / (1 + a * b) * beta_fn(1.0 / a, b)
    kl += 1.0 / (2 + a * b) * beta_fn(2.0 / a, b)
    kl += 1.0 / (3 + a * b) * beta_fn(3.0 / a, b)
    kl += 1.0 / (4 + a * b) * beta_fn(4.0 / a, b)
    kl += 1.0 / (5 + a * b) * beta_fn(5.0 / a, b)
    kl += 1.0 / (6 + a * b) * beta_fn(6.0 / a, b)
    kl += 1.0 / (7 + a * b) * beta_fn(7.0 / a, b)
    kl += 1.0 / (8 + a * b) * beta_fn(8.0 / a, b)
    kl += 1.0 / (9 + a * b) * beta_fn(9.0 / a, b)
    kl += 1.0 / (10 + a * b) * beta_fn(10.0 / a, b)
    kl *= (prior_beta - 1.0) * b

    # use another taylor approx for Digamma function
    # psi_b_taylor_approx = torch.log(b) - 1. / (2 * b) - 1. / (12 * b ** 2)
    psi_b = torch.digamma(b)

    kl += (a - prior_alpha) / a * (-EULER_GAMMA - psi_b - 1.0 / b)  # T.psi(self.posterior_b)

    # add normalization constants
    kl += torch.log(a + EPSILON) + torch.log(b + EPSILON) + torch.log(beta_fn(prior_alpha, prior_beta) + EPSILON)

    # final term
    kl += -(b - 1) / b

    skl = kl.sum(dim=-1)
    if reduction == "mean":
        return torch.mean(skl)
    elif reduction == "sum":
        return torch.sum(skl)
    return skl

def kl_bernoulli(p, q):
    A = p * (torch.log(p + EPSILON) - torch.log(q + EPSILON))
    B = (1.0 - p) * (torch.log(1.0 - p + EPSILON) - torch.log(1.0 - q + EPSILON))
    KL = A + B
    return KL.sum()

# ====================================================================
# DEEP KALMAN FILTER ENCODERS
# ====================================================================

class q_INDEP(nn.Module):
    input_dim: int
    par_1: nn.Module
    par_2: nn.Module
    q_infer: nn.Module
    data_2_hidden: MLP
    distribution_type: str

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim: int = kwargs.get("control_variable_dim", 0)
        self.observable_dim: int = kwargs.get("observable_dim")
        self.layers_dim: int = kwargs.get("layers_dim")
        self.hidden_state_dim: int = kwargs.get("hidden_state_dim")
        self.out_dim: int = kwargs.get("output_dim")
        self.dropout: float = kwargs.get("dropout", 0.0)
        self.out_droupout: float = kwargs.get("out_dropout")
        self.normalization: bool = kwargs.get("normalization")
        self.distribution_type: float = kwargs.get("distribution_type", "normal")
        self.alpha0: int = kwargs.get("alpha0")
        self.define_deep_models_parameters()

    def define_deep_models_parameters(self):
        self.input_dim = self.control_variable_dim + self.observable_dim
        self.data_2_hidden = MLP(
            input_dim=self.input_dim,
            layers_dim=self.layers_dim,
            output_dim=self.out_dim,
            output_transformation=None,
            normalization=self.normalization,
            dropout=self.dropout,
        )
        if self.out_droupout > 0:
            self.out_dropout_layer = torch.nn.Dropout(self.out_droupout)
        self.par_1 = nn.Linear(self.out_dim, self.hidden_state_dim)
        self.par_2 = nn.Linear(self.out_dim, self.hidden_state_dim)
        if self.distribution_type == NORMAL:
            self.q_infer = self._q_normal
        elif self.distribution_type == BERNOULLI:
            self.q_infer = self._q_bernoulli
        elif self.distribution_type == KUMARASWAMY_BETA:
            self.q_infer = self._q_kumaraswamy
        else:
            raise ValueError(f"Invalid distribution type {self.distribution_type}!")

    def forward(self, data, x=None, prior_transform=None):
        """
        data
        ---------
        batch_size,seq_length,observables_size

        parameters
        ---------
        data (batch_size,seq_length,observables_dimension)

        returns
        -------
        z,(z_mean, z_sigma)

        """


        if x is not None:
            data = torch.cat((data, x), dim=-1).unsqueeze(1)
        else:
            x = torch.zeros(data.size(0), self.hidden_state_dim, device=self.device)

        h = self.data_2_hidden(data).to(torch.float32)
        h = self.out_dropout_layer(h) if self.out_droupout > 0 else h

        if prior_transform is not None:
            x = prior_transform(x)

        return self.q_infer(h, x)

    def _q_bernoulli(self, h, x, batch_size, seq_length):
        posterior_pi_logit = self.par_1(h).view(batch_size * seq_length, -1)
        posterior_pi = torch.sigmoid(posterior_pi_logit)
        prior_pi = torch.cumprod(x, dim=-1)

        kl = kl_bernoulli(posterior_pi, prior_pi)
        pi_ = posterior_pi.view(-1).unsqueeze(1)  # [batch_size*number_of_topics,1]
        pi_ = torch.cat((pi_, 1.0 - pi_), dim=1)
        b = gumbel_softmax(pi_, 1.0, self.device)[:, 0]  # [batch_size*number_of_topics]
        b = b.view(batch_size, -1)  # [batch_size,number_of_topics]
        return b, posterior_pi, kl

    def _q_normal(self, h, x):
        mu = self.par_1(h)
        logvar = self.par_2(h)
        z = reparameterize_normal(mu, logvar)
        kl = kullback_leibler_two_gaussians(mu, logvar, x, torch.zeros_like(x), "sum")
        return z, (mu, logvar), kl

    def _q_kumaraswamy(self, h, x, batch_size, seq_length):
        one = torch.tensor(1.0, device=self.device)
        a = torch.nn.functional.softplus(self.par_1(h).view(batch_size * seq_length, -1))
        b = torch.nn.functional.softplus(self.par_2(h).view(batch_size * seq_length, -1))

        z = reparameterize_kumaraswamy(a, b)
        kl = kl_kumaraswamy_beta(a, b, torch.round(x * self.alpha0), one, "sum")
        return z, (a, b), kl

    def normal_(self):
        torch.nn.init.normal_(self.data_2_hidden)
        torch.nn.init.normal_(self.par_1)
        torch.nn.init.normal_(self.par_2)

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    @classmethod
    def sample_model_parameters(self):
        parameters = {"control_variable_dim": 3,
                      "observable_dim": 3, "layers_dim": [10],
                      "hidden_state_dim": 3}
        return parameters

    @classmethod
    def get_parameters(cls):
        kwargs = {
            "observable_dim": 10,
            "layers_dim": [250, 250],
            "output_dim": 250,
            "hidden_state_dim": 32,
            "dropout": 0.1,
            "out_dropout": 0.1,
        }
        return kwargs

    def init_parameters(self):
        return None

class q_RNN(nn.Module):
    input_dim: int
    data_2_hidden: nn.LSTM
    meanmu: nn.Module
    log_var: nn.Module
    is_past_state: bool
    data_2_latent: MLP

    def __init__(self, device=None, **kwargs):
        nn.Module.__init__(self)
        if device is None:
            self.device = torch.device("cpu")

        self.control_variable_dim = kwargs.get("control_variable_dim", 0)
        self.observable_dim = kwargs.get("observable_dim")
        self.layers_dim = kwargs.get("layers_dim")
        self.hidden_state_dim = kwargs.get("hidden_state_dim")
        self.hidden_state_transition_dim = kwargs.get("hidden_state_transition_dim", 10)
        self.n_rnn_layers = kwargs.get("num_rnn_layers", 1)
        self.dropout = kwargs.get("dropout", 0.4)
        self.out_droupout: float = kwargs.get("out_dropout")
        self.delta = kwargs.get("delta", 0.005)

        self.define_deep_models_parameters()

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, value):
        self.__device = value
        self.to(value)

    def define_deep_models_parameters(self):
        # RECOGNITION MODEL
        self.input_dim = self.control_variable_dim + self.observable_dim
        if self.layers_dim is not None:
            self.data_2_latent = nn.Linear(self.input_dim, self.layers_dim)
            self.input_dim = self.layers_dim
        if self.out_droupout > 0:
            self.out_dropout_layer = torch.nn.Dropout(self.out_droupout)
        self.data_2_hidden = nn.LSTM(
            self.input_dim, self.hidden_state_transition_dim, dropout=self.dropout, batch_first=True, num_layers=self.n_rnn_layers
        )
        self.mean = nn.Linear(self.hidden_state_transition_dim + self.hidden_state_dim, self.hidden_state_dim)
        self.log_var = nn.Linear(self.hidden_state_transition_dim + self.hidden_state_dim, self.hidden_state_dim)

    def init_hidden_rnn_state(self, batch_size):
        hidden_init = (
            torch.randn(self.n_rnn_layers, batch_size, self.hidden_state_transition_dim, device=self.device),
            torch.randn(self.n_rnn_layers, batch_size, self.hidden_state_transition_dim, device=self.device),
        )

        return hidden_init

    def forward(self, data, prior=None, prior_data=None):
        """

        :param data: (B, L, D)
        :return:
        """
        batch_size, seq_length, _ = data.shape
        if self.layers_dim is not None:
            data = self.data_2_latent(data)
        h = self.init_hidden_rnn_state(batch_size)
        out, _ = self.data_2_hidden(data, h)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))
        out = out.reshape(batch_size * seq_length, -1)
        out = self.out_dropout_layer(out) if self.out_droupout > 0 else out

        z = torch.zeros(seq_length, self.hidden_state_dim, device=self.device)
        m_q = torch.zeros(seq_length - 1, self.hidden_state_dim, device=self.device)
        logvar_q = torch.zeros(seq_length - 1, self.hidden_state_dim, device=self.device)

        in_0 = torch.cat([out[0], torch.zeros(self.hidden_state_dim, device=self.device)])
        mu_0: torch.Tensor = self.mean(in_0)
        logvar_0 = self.log_var(in_0)
        z[0] = reparameterize_normal(mu_0, logvar_0)
        assert not torch.any(torch.isnan(z))
        assert not torch.any(torch.isinf(z))
        logvar_t_p = torch.log(self.delta * torch.ones_like(logvar_0))

        kl = kullback_leibler(mu_0, logvar_0, "sum")

        for t in range(1, seq_length):
            in_t = torch.cat([out[t], z[t - 1]])
            mu_t: torch.Tensor = self.mean(in_t)
            logvar_t = self.log_var(in_t)
            m_q[t - 1] = mu_t
            logvar_q[t - 1] = logvar_t
            z[t] = reparameterize_normal(mu_t, logvar_t)
            if prior is None:
                mu_t_p = z[t - 1]
                kl += kullback_leibler_two_gaussians(mu_t, logvar_t, mu_t_p, logvar_t_p, "sum")
            assert not torch.any(torch.isnan(z))
            assert not torch.any(torch.isinf(z))
        if prior is not None:
            if prior_data is None:
                p_m = prior(z[:-1])
            else:
                p_m = prior(torch.cat((z[:-1], prior_data[:-1]), dim=-1))
            kl += kullback_leibler_two_gaussians(m_q, logvar_q, p_m, torch.log(self.delta * torch.ones_like(z[:-1])), "sum")
        return z, (mu_t, logvar_t), kl

    @classmethod
    def sample_model_parameters(self):
        parameters = {
            "control_variable_dim": 3,
            "observable_dim": 3,
            "layers_dim": [10],
            "hidden_state_dim": 3,
            "hidden_state_transition_dim": 10,
        }
        return parameters

    def init_parameters(self):
        return None

    @classmethod
    def get_parameters(cls):
        kwargs =  {
            "observable_dim": 10,
            "layers_dim": 400,
            "num_rnn_layers": 4,
            "hidden_state_dim": 32,
            "hidden_state_transition_dim": 400,
            "dropout": 0.1,
            "out_dropout": 0.0,
        }
        return kwargs


if __name__=="__main__":
    from supervisedllm.data.dataloaders import TopicDataLoader
    from supervisedllm import data_path

    data_dir = data_path / "preprocessed" / "arxiv"
    dataloader_params = {"root_dir": data_dir, "batch_size": 32 ,"is_dynamic":True}
    data_loader = TopicDataLoader("cpu", **dataloader_params)
    databatch = next(data_loader.train.__iter__())

    recognition_parameters = q_INDEP.get_parameters()
    recognition_parameters.update({"observable_dim":data_loader.vocabulary_dim})

    q = q_INDEP(**recognition_parameters)
    corpus = databatch["corpus"]
    batch_size, seq_length, _ = corpus.shape
    results = q(corpus.view(batch_size * seq_length, -1).to(torch.float32))


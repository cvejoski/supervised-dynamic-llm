from pathlib import Path

base_path = Path(__file__).resolve().parent
project_path = (base_path / ".." / "..").resolve()
data_path = project_path / "data"
test_resources_path = project_path / "tests" / "resources"

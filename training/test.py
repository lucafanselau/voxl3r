


def test_run(run_name, project_name):
    run = wandb.init(project=project_name, name=run_name)
    run.log({"test": "test"})
    run.finish()

if __name__ == "__main__":
    project_name = "mast3r-3d-proxy"
    run_name = "2NHs8_1_2"
    test_run(run_name, project_name)
import os
import neptune
import config


class NeptuneLogger:
    """
    Контейнер для логгирования экспериментов с помощью ui.neptune.ai
    """

    def __init__(self, kwargs):
        super(NeptuneLogger,  self).__init__()
        self.kwargs = kwargs
        self.project_name = self.kwargs['project_name']
        self.params = self.kwargs['params']
        self.artifact_path = self.kwargs['artifact_path']
        self.image_path = self.kwargs['image_path']
        self.tag_list = self.kwargs['tag_list']

    def init_client(self):
        neptune.init(project_qualified_name=config.NEPTUNE_PROJECT_QUAL_NAME,
                     api_token=config.NEPTUNE_TOKEN)
        print(f'[NeptuneLogger] Initiate. Project: {self.project_name}')

    def create_experiment(self):
        neptune.create_experiment(name=f"""exp_{self.kwargs['training_data_sha1']}""",
                                  tags=self.tag_list,
                                  params=self.params)

    def log_metric(self, name, value):
        neptune.log_metric(name, value)

    def log_artifact(self, filename):
        neptune.log_artifact(os.path.join(self.artifact_path, filename))

    def log_image_artifact(self, filename):
        neptune.log_artifact(os.path.join(self.image_path, filename))

    def log_image_array(self, name, array):
        neptune.log_image(name, array)

    def log_text(self, log_name, text):
        if not isinstance(text, str):
            text = str(text)
        neptune.log_text(log_name, text)

    def finish_experiment(self):
        neptune.stop()




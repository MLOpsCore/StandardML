from standardml.runtime.executor import InputConfigExtractor, Executor
from standardml.runtime.parse import InputConfig


class WorkerApplication:

    def __init__(self, input_config: InputConfig):
        self.input_config: InputConfig = input_config
        self.extractor = InputConfigExtractor(input_config)

    def run(self):
        # TODO: Mount units
        executor: Executor = self.extractor.extract()
        executor.run()

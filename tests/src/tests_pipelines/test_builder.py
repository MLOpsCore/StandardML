from standardml.pipelines.builder import PipelineBuilder


def test_pipeline_basic():
    pipeline_tasks = 'resources/pipelines/pipeline_basic.json'
    pipeline = PipelineBuilder.build(pipeline_tasks=pipeline_tasks)
    result = pipeline.execute(0)

    assert result == 100


def test_pipeline_image():
    image = 'resources/datasets/imgimg_black/image.png'
    pipeline_tasks = 'resources/pipelines/pipeline_image_input_and_label.json'
    pipeline_input, pipeline_label = PipelineBuilder.build(pipeline_tasks=pipeline_tasks)

    result_input = pipeline_input.execute(image)
    result_label = pipeline_label.execute(image)

    assert result_input.mean() == 1.0 and result_label.mean() == 1.0
    assert result_input.shape == (128, 128, 3) and result_label.shape == (128, 128, 1)

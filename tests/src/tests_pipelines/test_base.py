from standardml.pipelines.base import Pipeline, PipelineTask


def test_simple_pipeline():
    pipeline: Pipeline = Pipeline()

    pipeline.add(
        PipelineTask(msg='no-added', funct=lambda x: x, args={}), condition=False)

    pipeline.extend([
        PipelineTask(msg='task-1', funct=lambda x: x + 1, args={}),
        PipelineTask(msg='task-2', funct=lambda x: x + 1, args={}),
        PipelineTask(msg='task-3', funct=lambda x: x + 1, args={}),
    ])

    pipeline.add(PipelineTask(
        msg='multiply', funct=lambda x, another: x * 100, args=dict(another=100)))

    result = pipeline.execute(0)

    assert pipeline.tasks[0].msg != 'no-added', 'Pipeline with msg "no-added" should not be added'
    assert len(pipeline.tasks) == 4, 'Pipeline should have only 3 tasks'
    assert result == 300, 'Pipeline result is not correct'

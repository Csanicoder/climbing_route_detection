# Generate Summary Data DSL

Prerequisite:
``` bash
pip install datamodel-code-generator 0.45.0
```

Summary DSL gen:

``` bash
datamodel-codegen \
 --input ../json_schemas/summary.schema.json \
 --input-file-type jsonschema \
 --output summary.py \
 --class-name SummaryData
```

Frame Analytics DSL gen:

``` bash
datamodel-codegen \
 --input ../json_schemas/frame_analytics.schema.json \
 --input-file-type jsonschema \
 --output frame_analytics.py \
 --class-name FrameAnalyticsData \
 --output-model-type pydantic_v2.BaseModel \
 --collapse-root-models
```


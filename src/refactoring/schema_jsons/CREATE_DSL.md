# Generate Summary Data DSL

Prerequisite:
``` bash
pip install datamodel-code-generator 0.45.0
```

Hold Data DSL gen:
``` bash
cd src/refactoring/schema_jsons \
&& \
datamodel-codegen \
 --input hold.schema.json \
 --input-file-type jsonschema \
 --output ../data/models/hold.py \
 --class-name HoldData \
 --output-model-type pydantic_v2.BaseModel \
 --collapse-root-models
```

Pose Data DSL gen:
``` bash
cd src/refactoring/schema_jsons \
&& \
datamodel-codegen \
 --input pose.schema.json \
 --input-file-type jsonschema \
 --output ../data/models/pose.py \
 --class-name PoseData \
 --output-model-type pydantic_v2.BaseModel \
 --collapse-root-models
```

Summary DSL gen:

``` bash
cd src/refactoring/schema_jsons \
&& \
datamodel-codegen \
 --input summary.schema.json \
 --input-file-type jsonschema \
 --output ../data/models/summary.py \
 --class-name SummaryData
```

Frame Analytics DSL gen:

``` bash
cd src/refactoring/schema_jsons \
&& \
datamodel-codegen \
 --input frame_analytics.schema.json \
 --input-file-type jsonschema \
 --output ../data/models/frame_analytics.py \
 --class-name FrameAnalyticsData \
 --output-model-type pydantic_v2.BaseModel \
 --collapse-root-models
```


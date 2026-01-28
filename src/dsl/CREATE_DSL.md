# Generate Summary Data DSL

Prerequisite:
``` bash
pip install datamodel-code-generator 0.45.0
```

``` bash
datamodel-codegen \
 --input ../json_schemas/summary.schema.json \
 --input-file-type jsonschema \
 --output summary.py \
 --class-name SummaryData
```
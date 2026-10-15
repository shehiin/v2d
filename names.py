"""Map rollout body labels e<N>_<object> back to the roster's <object>, in place.

tr.sh prefixes body names with the episode only so URDFs of the same object from different
episodes don't overwrite each other in one folder; the packer wants the roster names.
Schema metadata (evaluator and checkpoint hashes) is kept as is.

    python names.py <episode> <rollout parquet>
"""
import sys

import pyarrow as pa
import pyarrow.parquet as pq

episode, path = sys.argv[1], sys.argv[2]
t = pq.read_table(path)
prefix = f"e{episode}_"
col = t["object_body_name"].combine_chunks()
names = [n[len(prefix):] if n.startswith(prefix) else n for n in col.dictionary.to_pylist()]
fixed = pa.DictionaryArray.from_arrays(col.indices, pa.array(names, pa.string()))
i = t.schema.get_field_index("object_body_name")
t = t.set_column(i, t.schema.field(i).with_type(fixed.type), fixed)
pq.write_table(t.replace_schema_metadata(t.schema.metadata), path)
print(path, sorted(set(names)))

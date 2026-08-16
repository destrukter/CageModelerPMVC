# Evaluation configs

`gen_eval_configs.py` generates the evaluation JSON files an evaluation run consumes. It
is a *generator*, not a runner: it never loads a mesh, computes coordinates or exports a
result. Only the standard library and numpy are imported, so it runs standalone without a
build of the viewer.

## Workflow

1. Edit the `CONFIGURATION` block at the top of `gen_eval_configs.py`.
2. Generate:

   ```sh
   python3 evaluation/gen_eval_configs.py
   ```

   It writes one config per (model entry x coordinate setup) pair into
   `evaluation/generated/`, plus a `manifest.json` listing all of them.

3. Run the whole batch in one launch by pointing at the directory:

   ```sh
   ./CageModeler --eval-config evaluation/generated
   ```

   A directory runs every `*.json` in it, in sorted order, skipping `manifest.json`. Each
   config keeps its own timings file and its own output directory, exactly as when run on
   its own, so a batch and a series of single runs produce the same files.

   To run a subset instead, name the configs — the flag may be repeated:

   ```sh
   ./CageModeler --eval-config evaluation/generated/<a>.json \
                 --eval-config evaluation/generated/<b>.json
   ```

   The environment variable takes the same values, several separated by the platform's
   path list separator (`;` on Windows, `:` elsewhere):

   ```sh
   CAGEMODELER_EVAL_CONFIG=evaluation/generated ./CageModeler
   ```

   Without an override the viewer falls back to `evaluation/projects.json`.

Use `--dry-run` to validate the configuration without writing anything.

### Working directory

The viewer resolves `evaluation` relative to its working directory, and the mesh paths
inside a config are relative to that `evaluation` directory in turn. It therefore has to
run from the directory holding both `assets/` and `evaluation/`, which is where CMake puts
the executable and the packaged meshes — call it the run directory:

```
build/<preset>/bin/viewer/app/
```

The build does not put an `evaluation` directory there, so a generated config has to be
reachable from it. The simplest way is to generate straight into the run directory, which
needs no symlink and no special privileges:

```sh
python3 evaluation/gen_eval_configs.py \
    --output-dir build/<preset>/bin/viewer/app/evaluation/generated
```

`--output-dir` and `--asset-root` are relative to the current working directory, unlike
the `OUTPUT_DIR` and `ASSET_ROOT` constants, which are relative to the script.

Alternatively link the source `evaluation` directory into the run directory once, and
regenerate normally from then on:

```sh
# Linux / macOS
cd build/<preset>/bin/viewer/app
ln -s ../../../../../evaluation evaluation
```

```bat
REM Windows, a junction needs no administrator rights
cd build\<preset>\bin\viewer\app
mklink /J evaluation ..\..\..\..\..\evaluation
```

The evaluation runs during startup, so launching the viewer with a config in place starts
it immediately. A missing config is logged and skipped rather than being an error, so
`does not exist. Skipping evaluation run.` means the path did not resolve — check that the
run directory really contains `evaluation\generated`, and that the filename matches one
the generator actually produced (`manifest.json` lists them all).

## Configuration

**Model entries** are `(model, cage, deformed cage)` triples. A single entry lists several
`deformed_cages`, so a model is never repeated. The three color map exports are properties
of the *entry*, because their vertex indices address the undeformed cage that all of the
entry's deformed cages share:

| Map                   | Toggle                   | Vertex selection                     |
| --------------------- | ------------------------ | ------------------------------------ |
| Influence map         | `influence_map`          | `influence_vertices` — a list        |
| Euclidean distance map| `euclidean_distance_map` | `euclidean_vertex` — exactly one     |
| Interior distance map | `interior_distance_map`  | `interior_vertex` — exactly one      |

Indices are 0-based into the undeformed cage: index *n* is the (*n*+1)-th `v` line of the
cage OBJ, because the cage loader keeps the file order.

**Coordinate setups** are a separate list of variants and their parameters
(`coordinate_type`, `hit_count`, `alpha` / `beta` / `theta`, `use_interior_distance`,
`samples`, and the distance map isoline settings). The generator emits the full cartesian
product of model entries and coordinate setups.

Filenames and project names are `<entry>__<deformed cage stem>__<setup>`, iteration is
sorted, so regenerating an unchanged configuration reproduces byte-identical output.

## Validation

Nothing is written until the whole configuration validates. Every problem found is
reported at once, naming the offending entry:

- every referenced file exists (and is not an unfetched Git LFS pointer)
- vertex indices are within the cage's vertex count and non-negative
- no duplicate model entry names, coordinate setup names or generated filenames
- exactly one vertex where exactly one is required, at least one where a list is required
- coordinate types are known, and a setup does not ask for something the viewer overrides
  (`PMVCO` forces a single hit and no interior distance)
- coordinate types requiring a tetrahedral embedding have one

## Paths

`ASSET_ROOT` is where the meshes live on disk and is used only for validation.
`CONFIG_PATH_PREFIX` is what gets written into the JSON. They differ because the viewer
resolves config paths relative to the `evaluation` directory at run time, against the
build directory's `assets/meshes`, which CMake packages from `models/`.

Validation reads the cage files, so fetch them first:

```sh
git lfs pull
```

## The interior distance map

The interior distances are read back from the table the interior-distance PMVC variant
fills during the run; they are never recomputed by the export. A coordinate setup that is
not PMVC with `use_interior_distance=True` therefore has nothing to read. The generator
still emits the pair, but writes `interiorDistanceMap: false` and records the reason in
the manifest, so no config silently exports nothing.

## Output

Per config, below the `evaluation` directory:

```
results/timings_<stem>.json                     one row per project: status, stage timings, vertex counts
results/<stem>/weights.dmat                     the computed weights
results/<stem>/deformed_cage.obj
results/<stem>/deformed_mesh.obj
results/<stem>/influence_map.obj                vertex coloured, if influenceMap is on
results/<stem>/influence_map_vertices.txt       the selection it was exported for
results/<stem>/euclidean_distance_map.obj       if euclideanDistanceMap is on
results/<stem>/interior_distance_map.obj        if interiorDistanceMap is on
```

Each color map is written next to a `_vertices.txt` listing the cage vertices it was
exported for, so an export can be traced back to its selection.

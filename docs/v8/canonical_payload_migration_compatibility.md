# Issue #259 Canonical Payload Migration and Compatibility

## Decision

`bigan-v8-canonical-payload-v1` is an additive semantic serialization contract.
It does not replace or rewrite any historical raw payload, JSONL row, manifest,
or raw SHA-256. New hash-bound decision artifacts retain their original bytes
and raw hash and add a canonical UTF-8 payload plus canonical SHA-256.

## New lineages

New frozen-versus-settled comparisons must:

1. bind both payloads to the same payload schema version;
2. independently validate and canonicalize both payloads;
3. require exact canonical SHA-256 equality;
4. require approved raw-artifact/source lineage;
5. emit the deterministic semantic diff when hashes differ;
6. fail closed for settlement and evaluation eligibility on any error.

The reusable implementation is `bigan.v8.canonical_payload`. The committed
contract and cross-runtime fixtures are immutable review artifacts.

## Legacy lineages

Legacy artifacts remain byte-for-byte immutable. Their existing raw-file and
raw-row verification continues unchanged. A legacy verification call must be
explicitly identified as legacy and must not synthesize canonical hashes into a
historical manifest. Canonical descriptors may be written only to a new
additive audit artifact.

No historical manifest is migrated in place. No old digest is reinterpreted as
a canonical digest. A raw digest and a canonical digest always use distinct
field names.

## Compatibility boundary

Canonical equality removes only incidental JSON representation differences:
object insertion order, whitespace, equivalent permitted number spelling,
Unicode composition, and negative zero. It does not drop fields, sort lists,
coerce strings to numbers, tolerate numeric drift, accept invalid timestamps,
or equate a missing field with null.

The settlement fallback still requires approved source lineage. Canonical hash
equality alone never grants training, export, paper, promotion, live, write,
wallet, capital, handoff, source-change, freeze-change, #134, or #146
eligibility.

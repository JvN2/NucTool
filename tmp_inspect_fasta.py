from pathlib import Path
from Bio import SeqIO

fasta_path = Path(".genomes/sacCer3/sacCer3.fa")
if not fasta_path.exists():
    print("FASTA not found at", fasta_path.resolve())
    alt = Path(r"C:\Users\jvann\Downloads\chrII.fa")
    if alt.exists():
        print("Found alt FASTA at", alt)
        fasta_path = alt
    else:
        print("No FASTA file available to inspect.")
        raise SystemExit(1)

print("Using FASTA:", fasta_path.resolve())
# Print first 20 headers
print("Scanning FASTA headers (first 20):")
for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
    print(i, rec.id, rec.description[:200])
    if i >= 19:
        break

# Broad pattern match search for chr II
patterns = [
    "chromosome ii",
    "chr ii",
    "chromosome 2",
    "chr2",
    "chr_2",
    "chrii",
    "chromosome_ii",
    "sc_chr_ii",
    "sc_chr2",
    "chromosome-ii",
]
found = []
for i, rec in enumerate(SeqIO.parse(fasta_path, "fasta")):
    desc = (rec.id + " " + rec.description).lower()
    for p in patterns:
        if p in desc:
            found.append((i, rec.id, p))
            break

print("\nPattern matches:")
if not found:
    print("No pattern matches found for chr II")
else:
    for f in found:
        print(f)

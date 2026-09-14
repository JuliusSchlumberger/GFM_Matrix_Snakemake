from remotezip import RemoteZip

url = "https://data.4tu.nl/file/1da2e70f-6c4d-4b03-86bd-b53e789cc629/22ffa027-184b-4f67-9979-c182f3dfb1ab"

with RemoteZip(url) as zf:
    # find the exact filename first (handles v1_0 vs v1_1 naming without guessing)
    matches = [n for n in zf.namelist() if "S25E033" in n]
    print("Found:", matches)

    for name in matches:
        print(f"Extracting {name} ...")
        zf.extract(name, path=".")

print("Done.")
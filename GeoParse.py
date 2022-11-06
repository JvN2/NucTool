import GEOparse

geo = 'GSE132225'
geo = 'GSM4193330'

gse = GEOparse.get_GEO(geo, destdir="./data")

for key, value in gse.metadata.items():
    print(key, value)

print(gse)
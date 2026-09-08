import requests
q = ("https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
     "USA_Flood_Hazard_Reduced_Set_gdb/FeatureServer/0/query")
r = requests.get(q, params={"where": "1=1", "returnCountOnly": "true", "f": "json"}, timeout=60)
print(r.status_code, r.json())
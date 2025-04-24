**<h1 align="center">ArchiMed Python Connector</h1>**

<p align="center">
  <img src="https://www.cic-it-nancy.fr/uploads/2024/10/logo_platform_archimed.png" alt="ArchiMed Logo" width="300">
</p>

## About ArchiMed
https://www.cic-it-nancy.fr/fr/plateforme-archimed/

## Imports
```python
from ArchiMedConnector.A3_Connector import A3_Connector # import Connector

from pprint import pprint  # import pprint to display json structures
```

## Create Connector Instance
```python
a3conn= A3_Connector()
help(a3conn)

# Get your user infos
help(a3conn.getUserInfos)
me = a3conn.getUserInfos()
print(me)
```

## Get A3 objects definition (useful for custom querying)
```python
help(a3conn.getNodeTypeSpecifications)
obj_def = a3conn.getNodeTypeSpecifications(
   "<OBJECT TYPE : 'STUDY' | 'WORKLIST' | 'TMPSTORAGE' | 'EXAM' | 'SERIE' | 'FILE'>" # Object Type
)
print("Obj Def :"); pprint(obj_def)
```

## Get Exam Information
```python
help(a3conn.getExamFullInfos)
examinfos1 = a3conn.getExamFullInfos(
   "<EXAM CODE>",  # Exam Code
    filterStr="<OPTIONAL FILTERING>",  # Optional filtering string (default empty) : ex. : "exam.examDescription like '%blzllr%' "
    orderbyStr="<OPTIONAL SORT>",  # Optional order by string (default empty) : ex. : "exam.examDate desc" or "file.DCM_MR.instanceNumber"
    worklistType="<OPTIONAL WORKLIST TYPE : '' | 'SPECIFIED' | 'BOTH' | 'EXAM' | 'SERIE'>",  # Optional worklist type string (default empty : not in worklist)
    specifiedWorklistName="<OPTIONAL WORKLIST NAME>"  # If worklist type = 'SPECIFIED', put here the name of the worklist
)
```

## Get Serie Information
```python
help(a3conn.getSerieFullInfos)
serieNumber = 3
serieinfos1 = a3conn.getSerieFullInfos(
   "<EXAM CODE>",  # Exam Code
    serieNumber,  # Serie Number or -1 (for serie's unclassified file)
    filterStr="<OPTIONAL FILTERING>",  # Optional filtering string (default empty) : ex. : "exam.examDescription like '%blzllr%' "
    orderbyStr="<OPTIONAL SORT>",  # Optional order by string (default empty) : ex. : "exam.examDate desc" or "file.DCM_MR.instanceNumber"
    worklistType="<OPTIONAL WORKLIST TYPE : '' | 'SPECIFIED' | 'BOTH' | 'EXAM' | 'SERIE'>",  # Optional worklist type string (default empty : not in worklist)
    specifiedWorklistName="<OPTIONAL WORKLIST NAME>"  # If worklist type = 'SPECIFIED', put here the name of the worklist
)
```

## Download files
```python
help(a3conn.downloadFiles)
a3conn.downloadFiles(
    serieinfos1,  # infos node (from root)
    asStream=False,  # optional output type (default False) if asStream=True, return IOBytes list, else return downloaded files paths
    destDir="./<OPTIONAL DEST DIR>",  # optional output dir (default '.'). If asStream=False, files download directory path
    fileTypeAcronym="<OPTIONAL FILETYPE ACRONYM>"  # Optional FileTypeAcronym to Download (default empty). ex. 'DCM', 'DCM_MR'
)

help(a3conn.downloadFile)
fileID = 999999999
a3conn.downloadFile(
    fileID,  # file id (see node infos)
    asStream=False,   # optional output type (default False) if asStream=True, return IOBytes ref, else return downloaded file path
    filePath="./<OPTIONAL FILE PATH>",  # optional output File path (default 'out'). If asStream=False, file path
    inWorklist=False  # Optional flag = True if file is in a worklist (default False).
)
```

##  Load Dicoms In Memory
```python
help(archimedreader.readDicomFromArchiMedInfos)
```

##  Load all Dicoms data into infos node. headers= list of headers, array= 3D numpy array
```python
headers, array = archimedreader.readDicomFromArchiMedInfos(
    a3conn, #  ArchiMed 3.2 Connector Object
    serieinfos1, # infos node (from root)
)
```

# UPLOAD FILE INTO TMP ZONES
```python
tmpstoragesinfos = a3conn.getTmpStorages()  # Get Authorized Tmp Storages infos (ID is mandatory for files upload)
print("Tmp Storages :"); pprint(tmpstoragesinfos)
```

## Build list of files to upload
```python
files = []
files.append("<PATH TO FILE 1>")
files.append("<PATH TO FILE 2>")
files.append("<PATH TO FILE 3>")
```

## Upload files
```python
tmpstorageid = 1
a3conn.uploadFiles(
    tmpstorageid,  # tmp storage zone ID
    files  #  files paths list
)
```

# GENERATE DICOM TOOLS

## Generate uid
```python
uid = a3conn.generateDicomUID()
print("new uid : ", uid)
```
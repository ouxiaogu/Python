# tacx 

TAC(Tachyon Application Center) jobs, data, and etc.

- For Base Job class, 'Job', it's independent to TACX GUI python api, user can use it without the environment of tflex_python.

- For FEM+ job classes, it's based on the TACX FEM+ GUI python api

- For MXP job classes, it's independent to the GUI python api currently, user can run MxpJob without the environment of tflex_python.

## class heriachy


* `Job`, base class in TachyonJob.py, for all job types, include the processing of jobinfo xml, job status, job submit by script, and etc.

* `femplusAPI`, various types of FEM+ python wrapper classes, core classes:
    - `FEMJob`, derived class, inherit from `Job` class in ./common/TachyonJob.py, can process tacx job setup, review, submit, and even the completed job result data processing, updating, add processes.
    - `ADICalJob`, `CheckJob`: derived classes of `FEMJob`, which are Job type based Classes

* `MxpJob`: derived class, inherit from `Job` class in ./common/TachyonJob.py
* `MxpStage`: two classes inside
    - `MxpStage`: read inxml, call stage::run() function, and save outxml by stage::save() function
    - `MxpStageXmlParser`: parser class for mxp inxml("iccfs"), outxml("occfs"), summary ("osumccfs") and summary kpis("osumkpis")

* `SEMContour`: MXP job data level, the MXP SEM contour class for unencrypted files, easily convert to pandas DataFrame and verse vice for python level data analysis
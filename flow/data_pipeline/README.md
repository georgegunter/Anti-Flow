## Useful Commands

Use the following command inside `flow/examples`, or specify the path for `simulate.py`

* To run a simulation with output stored locally only:

    ```python
    python simulate.py EXP_CONFIG --gen_emission
    ```

* To run a simulation and upload output to pipeline:

    ```python
    python simulate.py EXP_CONFIG --to_aws
    ```

* To run a simulation, upload output to pipeline, and mark it as baseline:

    ```python
    python simulate.py EXP_CONFIG --to_aws --is_baseline
    ```

* To run a simulation, upload output to pipeline, and mark it as benchmark:

    ```python
    python simulate.py EXP_CONFIG --to_aws --is_benchmark
    ```
    Submissions that are marked as benchmark will not affect the top scores table.


Use the following command inside `flow/flow/data_pipeline`, or specify the path

* In case you want to delete an entire submission, use the following command,

    ```python
    python run_query.py --delete_submission SOURCE_ID
    ```

    An confirmation request will prompt on commandline, enter `Y`/`Yes`/`y`/`yes` to continue,
    or enter anything else to abort. Please note that this process is **irreversible**,
    make sure you have a local copy of your data. The source_id is a unique id for each simulation
    in the formate of "flow_{some random hashcode}. Source_id will be printed when you run the simulation,
    or you can find it in the file name of the local trajectory file.

* In case you want to upload an submission manually
(for instance, the automatic upload process abort due to network error),
use the follwing command,

    ```python
    python run_query.py --upload EMISSION_DATA_PATH METADATA_PATH
    ```

    For the EMISSION_DATA_PATH, please use the one outputed by the simulation with the source id inside the file name.
    METADATA_PATH is the path the metadata file. This file should be outputted along with the emission file and have the source_id
    and the word METADATA in the file name.

* If you enter the wrong metadata and would like to fix it manually, please navigate to the S3 bucket `circles.data.pipeline`.
Inside the bucket, open the folder `metadata_table` and look for the folder with date of your submission. Please note that this
data uses UTC time. Inside the correct data folder, look for the folder with your submission's source_id.
Inside that folder, you can then download the CSV file and modify it locally. Please modify the metadata file using a text editor
instead of Office Excel becuase excel might change the format of some values and cause error in queries. After the modification,
simply upload the metadata file back to its origin location on S3 using its original name. To update your changes on the leaderboard, use the following command,

    ```python
    python run_query.py --update_leaderboard QUEUE_URL SOURCE_ID
    ```

    The QUEUE_URL is the URL of the SQS queue that handles query requests. For security concern, this URL is not listed here. You can find it by navigating to the SQS console. In the left drop down menu, click on `Queues`. In side `Queues`, look for the queue named `RunQueryRequests` and click on it. You should find the URL in the `Details` section of this queue.

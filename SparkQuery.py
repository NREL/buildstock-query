import json
import boto3
import logging
import time
import pandas as pd
import atexit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkQuery:
    def __init__(self, emr=None, s3=None, cluster_id=None, result_s3_full_path=None,
                 terminate_on_exit=False) -> None:
        """

        :param emr: boto3 emr client instance. If not provided, a new client instance will be created
        :param s3: boto3 s3 client instance. If not provided, a new client instance will be created.
        :param cluster_id: The cluster_id of the EMR cluster. If not provided, can't submit queries, but can still
                           query for cluster status or terminate clusters.
        :param result_s3_full_path: The path in s3 to store the query result. If not provided, cannot run queries.
        :param terminate_on_exit: If true, terminates the cluster when the python program exits. Set this to true if
                                  you have started the cluster to run a single set of analysis and you would like to
                                  terminate the cluster when it completes. You can always manually call the terminate_
                                  cluster function.
        """

        self.emr = boto3.client('emr', region_name='us-west-2') if emr is None else emr
        self.s3 = boto3.client('s3') if s3 is None else s3
        self.cluster_id = cluster_id
        self.result_s3_full_path = result_s3_full_path
        if terminate_on_exit:
            atexit.register(self.terminate_cluster)
        if cluster_id is None or result_s3_full_path is None:
            logger.warning("Until you set cluster_id and result_s3_path, you can't run queries. ")

    @property
    def result_s3_full_path(self):
        return self._result_s3_full_path

    @result_s3_full_path.setter
    def result_s3_full_path(self, value):
        if value is not None:
            self.result_s3_bucket, self.result_s3_path = self.get_bucket_and_path(value)
        self._result_s3_full_path = value

    def wait_for_cluster_to_start(self, cluster_id=None):
        cluster_id = self.cluster_id if cluster_id is None else cluster_id
        status = ''
        while True:
            status = self.emr.describe_cluster(ClusterId=self.cluster_id)['Cluster']['Status']['State']
            if status.lower() in ['waiting']:
                break
            logger.info(f"Cluster is {status}")
            time.sleep(30)

    def get_cluster_status(self, cluster_id=None):
        cluster_id = self.cluster_id if cluster_id is None else cluster_id
        return self.emr.describe_cluster(ClusterId=self.cluster_id)['Cluster']['Status']['State']

    def terminate_cluster(self, cluster_id=None):
        cluster_id = self.cluster_id if cluster_id is None else cluster_id
        logger.info(f"Terminating cluster: {cluster_id}")
        self.emr.terminate_job_flows(JobFlowIds=[cluster_id])
        logger.info(f"Terminated cluster: {cluster_id}")

    @staticmethod
    def get_bucket_and_path(s3_path):
        if not s3_path.lower().startswith('s3://') or len(s3_path.split('//')) != 2:
            raise ValueError(f"Path should have the format: s3://bucket/path/to/output. This is wrong: {s3_path}")

        bucket = s3_path.split('//')[1].split('/')[0]
        path = '/'.join(s3_path.split('//')[1].split('/')[1:])
        path = path[:-1] if path.endswith('/') else path
        return bucket, path

    def get_notebook_params(self, execution_id):
        result = self.emr.describe_notebook_execution(NotebookExecutionId=execution_id)
        params = json.loads(result['NotebookExecution']['NotebookParams'])
        return params

    def get_notebook_status_dict(self, execution_id):
        notebook_status = self.emr.describe_notebook_execution(NotebookExecutionId=execution_id)
        query_status = notebook_status['NotebookExecution']['Status'].upper()
        if query_status in ['RUNNING', 'STARTING']:
            return {'result': "RUNNING"}
        elif query_status in ['FINISHED', 'FINISHING']:
            params = self.get_notebook_params(execution_id)
            obj = self.s3.get_object(Bucket=params['result_s3_bucket'],
                                     Key=f"{params['result_s3_path']}/result.txt")
            result = obj['Body'].read()
            logger.info(result)
            result_json = json.loads(result)
            return result_json
        elif query_status in ['STOPPED', 'STOPPING']:
            return {'result': 'CANCELLED', 'error': 'User stopped the notebook execution'}
        else:
            if "OutputNotebookURI" in notebook_status['NotebookExecution']:
                output_notebook_path = notebook_status['NotebookExecution']['OutputNotebookURI']
            else:
                output_notebook_path = "Output Notebook not available."

            return {'result': "FAILED", "error": f"Something wrong in query execution; ended with {notebook_status}."
                                                 f" Check the output notebook at: {output_notebook_path}"}

    def get_query_status(self, execution_id):
        result_json = self.get_notebook_status_dict(execution_id)
        return result_json['result']

    def get_query_error(self, execution_id):
        result_json = self.get_notebook_status_dict(execution_id)
        return result_json["error"]

    def wait_execution_to_finish(self, execution_id, timeout_minutes=60):
        start_time = time.time()
        while time.time() - start_time < timeout_minutes * 60:
            result_json = self.get_notebook_status_dict(execution_id)
            if result_json['result'] in ['RUNNING']:
                logger.info(f"Spark Query status is {result_json['result']}.")
            else:
                return result_json
            time.sleep(30)
        raise TimeoutError(f"The query didn't complete even after {timeout_minutes} minutes.")

    def get_all_running_queries(self):
        executions = self.emr.list_notebook_executions()
        running_executions = []
        for res in executions['NotebookExecutions']:
            if res['Status'].upper() in ['RUNNING', 'STARTING']:
                running_executions.append(res['NotebookExecutionId'])
        return running_executions

    def stop_execution(self, execution_id):
        self.emr.stop_notebook_execution(NotebookExecutionId=execution_id)

    def get_query_result(self, execution_id, timeout_minutes=60):
        result_json = self.wait_execution_to_finish(execution_id, timeout_minutes=timeout_minutes)
        if result_json['result'].upper() == "SUCCEEDED":
            logger.info("Query SUCCEEDED. Reading output data.")
            df = pd.read_csv(result_json['output_path'])
            return df
        else:
            raise Exception(f"Query {result_json['result']} with error. {result_json['error']}.")

    def spark_query(self, athena_db, query,
                    run_async=False):
        params = {
            "athena_db": athena_db,
            "result_s3_bucket": self.result_s3_bucket,
            "result_s3_path": self.result_s3_path,
            "query": query
        }

        start_resp = self.emr.start_notebook_execution(
            EditorId='e-2WMHQWJ11EQSV6QQWZQTZZ08H',  # Known NotebookID for the enduse_query1 notebook
            RelativePath='athena_query.ipynb',
            ExecutionEngine={'Id': self.cluster_id},  # j-XXXXXXXXXX number of the cluster
            ServiceRole='EMR_Notebooks_DefaultRole',
            NotebookParams=json.dumps(params)

        )
        execution_id = start_resp["NotebookExecutionId"]
        logger.info(f"ExecutionID:{execution_id}")
        if run_async:
            return execution_id
        else:
            self.get_query_result(execution_id), execution_id


if __name__ == "__main__":
    # example query
    result_s3_path = "s3://eulp/emr/radhikar/test/test1"
    # Cluster ID can be found from the aws console
    mySpark = SparkQuery(cluster_id='j-2DMA12HKLAGWJ')
    print(mySpark.result_s3_full_path)
    mySpark.result_s3_full_path = result_s3_path

    query = """
    select * from "res_national_53_2018_timeseries" limit 100;
    """
    athena_db = 'enduse'
    res = mySpark.spark_query(athena_db, query, run_async=True)
    print(res)
    mySpark.stop_execution(res)

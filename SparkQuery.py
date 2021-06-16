import json
import boto3
import logging
import time
import pandas as pd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkQuery:
    def __init__(self, emr, s3, cluster_id, result_s3_path) -> None:
        self.emr = emr
        self.s3 = s3
        self.cluster_id = cluster_id
        # s3://eulp/emr/nmerket/
        self.result_s3_bucket, self.result_s3_path = self.get_bucket_and_path(result_s3_path)

    def wait_for_cluster_to_start(self):
        status = ''
        while True:
            status = self.emr.describe_cluster(ClusterId=self.cluster_id)['Cluster']['Status']['State']
            if status.lower() in ['waiting']:
                break
            logger.info(f"Cluster is {status}")
            time.sleep(30)

    def get_cluster_status(self):
        return self.emr.describe_cluster(ClusterId=self.cluster_id)['Cluster']['Status']['State']

    def terminate_cluster(self):
        self.emr.terminate_job_flows(JobFlowIds=[self.cluster_id])

    @staticmethod
    def get_bucket_and_path(s3_path):
        if not s3_path.lower().startswith('s3://'):
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
    emr = boto3.client('emr', region_name='us-west-2')
    s3 = boto3.client('s3')
    result_s3_path = "s3://eulp/emr/radhikar/test/test1"
    # Cluster ID can be found from the aws console
    mySpark = SparkQuery(emr, s3, cluster_id='j-2DMA12HKLAGWJ', result_s3_path=result_s3_path)
    query = """
    select * from "res_national_53_2018_timeseries" limit 100;
    """
    athena_db = 'enduse'
    res = mySpark.spark_query(athena_db, query, run_async=True)
    print(res)
    mySpark.stop_execution(res)

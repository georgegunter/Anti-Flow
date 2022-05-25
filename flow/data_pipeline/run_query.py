"""runner script for invoking query manually."""
import argparse
from flow.data_pipeline.data_pipeline import delete_table, upload_to_s3, rerun_query, \
                                             get_completed_queries, put_completed_queries
import boto3

import pandas as pd
from datetime import timezone
from datetime import datetime

parser = argparse.ArgumentParser(prog="run_query", description="runs query on AWS Athena and stores the result to"
                                                               "a S3 location")
parser.add_argument("--delete_submission", type=str, nargs='*')
parser.add_argument("--upload", type=str, nargs=2)
parser.add_argument("--update_leaderboard", type=str, nargs=2)


if __name__ == "__main__":
    args = parser.parse_args()
    s3 = boto3.client('s3')
    sqs = boto3.client('sqs')

    if args.delete_submission:
        for sid in args.delete_submission:
            delete_table(s3, only_query_result=False, source_id=sid)
    elif args.upload:
        trajectory_table_path = args.upload[0]
        source_id = trajectory_table_path.split('/')[-1].split('.')[0]
        metadata_table_path = args.upload[1]
        assert source_id in metadata_table_path, 'trajectory table and metadata table\
                                                 must belong to the same submission'
        metadata = pd.read_csv(metadata_table_path)
        cur_datetime = datetime.now(timezone.utc)
        cur_date = cur_datetime.date().isoformat()
        upload_to_s3(
                'circles.data.pipeline',
                'metadata_table/date={0}/partition_name={1}_METADATA/'
                '{1}_METADATA.csv'.format(cur_date, source_id),
                metadata_table_path
        )
        upload_to_s3(
            'circles.data.pipeline',
            'fact_vehicle_trace/date={0}/partition_name={1}/{1}.csv'.format(cur_date, source_id),
            trajectory_table_path,
            {'network': metadata['network'][0],
             'is_baseline': metadata['is_baseline'][0],
             'version': metadata['version'][0],
             'on_ramp': metadata['on_ramp'][0],
             'road_grade': metadata['road_grade'][0]}
        )
    elif args.update_leaderboard:
        source_id = args.update_leaderboard[0]
        queue_url = args.update_leaderboard[1]
        lambda_temp = get_completed_queries(s3, source_id)
        lambda_temp.remove("LEADERBOARD_CHART_AGG")
        put_completed_queries(s3, source_id, lambda_temp)
        rerun_query(s3, sqs, queue_url, table='leaderboard_chart_agg')

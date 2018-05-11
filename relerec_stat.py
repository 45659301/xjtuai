#!/bin/env python
#-*- coding:utf-8 -*-
import sys
import operator
import argparse
import datetime
import math
from bigflow_python.proto import sample_pb2
from bigflow import base, input, output
from bigflow import transforms
from bigflow import serde
task_name = __file__.split('/')[-1].split('.')[0]
"""input paths"""
local_input = "./samples/part-00000"
afs_input = "afs://tianqi.afs.baidu.com:9902/user/feed_video/haokan_sample"
"""output paths"""
local_output = "local_test_output/"
afs_output = "afs://tianqi.afs.baidu.com:9902/user/feed_video/user/chenhuimin"
"""intermediate results & logging"""
afs_tmp = "afs://tianqi.afs.baidu.com:9902/user/feed_video/user/chenhuimin/bigflow_tmp/"
"""job configurations"""
job_conf = {
    'mapred.job.priority' : 'HIGH',
    'mapred.job.map.capacity' : '4000'
}
def emit_features(sample):
    user_feature = sample.user_feature
    request_feature = sample.request_feature
    context_feature = sample.context
    
    uid = user_feature.uid # uint64
    cuid = user_feature.cuid # bytes
    ua = request_feature.sofa_ua
    if ua not in (15, 16, 6):
        return

    rec_type_list = map(lambda x: x.recommend_feature.retrieval_feature, context_feature)
    duration_list = map(lambda x: x.content_feature.sv_duration, context_feature)
    clk_list = map(lambda x: x.user_feedback.click, context_feature) # bool
    playlength_list = map(lambda x: x.user_feedback.duration_total, context_feature)
    
    for i in range(len(rec_type_list)):
        rec_type = rec_type_list[i].retrieval_feature.recommend_type
        duration = duration_list[i]
        clk = 1 if clk_list[i] is True else 0
        playlength = playlength_list[i]
        if playlength > 50000 or duration > 50000 or duration == 0:
            break
        comp_rate = 0
        if clk == 1:
            comp_rate = min(1, float(playlength) / duration)
        yield ['{}#ALL\t{}'.format(ua, cuid), [1, clk, playlength, duration, comp_rate]]
        yield ['{}#{}\t{}'.format(ua,rec_type, cuid), [1, clk, playlength, duration, comp_rate]]

def average(res):
    return res.reduce(lambda a, b: map(operator.add, a, b)).map(lambda x, c: [c] + x, res.count())
    
if __name__ == "__main__":
    today = datetime.date.today().strftime("%Y%m%d") 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, help="the date to exec", default=today)
    parser.add_argument("-t", help="is local test", action="store_true")
    args = parser.parse_args()
    DATE = args.d
    ISTEST = args.t
    # input & output
    if ISTEST:
        input_path = [local_input]
        output_path = local_output
    else:
        input_path = ['{}/{}/{}/part-{}*'.format(afs_input, DATE, '00', '00000')]
        output_path = afs_output 
    output_path = '{}/{}/{}'.format(output_path, task_name, DATE)
        
    # job config
    job_name = '{}_{}_{}_{}'.format("feed_development", "chenhuimin", task_name, DATE)
    #job_name = 'feed_production_day_relerec_state' + "_" + DATE 
    pipeline = base.Pipeline.create("local" if ISTEST else "DAGMR",
            job_name=job_name,
            tmp_data_path=afs_tmp,
            hadoop_job_conf=job_conf)
    # 核心任务逻辑
    pipeline.add_file("./bigflow_python/proto/sample_pb2.py", "./sample_pb2.py")
    
    # to run in local mode, run code below first, then read from local file
    #pipeline = base.Pipeline.create("DAGMR",
    #        job_name=job_name,
    #        tmp_data_path=afs_tmp,
    #        hadoop_job_conf=job_conf)
    #pbs = pipeline.read(input.SequenceFile(*input_path, serde=serde.StrSerde()))
    #pipeline.write(pbs, output.SequenceFile(output_path, serde=serde.StrSerde()))
    #pipeline.run()
    pbs = pipeline.read(input.SequenceFile(*input_path, serde=serde.ProtobufSerde(sample_pb2.Sample)))
    p = pbs.flat_map(emit_features)\
        .group_by(key_extractor=lambda x:x[0], value_extractor=lambda x:x[1])\
        .apply_values(transforms.reduce, lambda a,b: map(operator.add, a, b)).flatten()\
        .map(lambda x: [x[0], x[1] + [float(x[1][2]) / (x[1][1]) if x[1][1] > 0 else 0]])\
        .group_by(key_extractor=lambda x:x[0].split('\t')[0], value_extractor=lambda x:x[1])\
        .apply_values(average).flatten().map(lambda x: '\t'.join(x[0].split('#') + map(str, x[1])))

    # output
    pipeline.write(p, output.TextFile(output_path).partition(n=1))
    pipeline.run()
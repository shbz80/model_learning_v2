import numpy as np
import pickle

blocks_bnn_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_bnn.p"
blocks_mgp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_blocks_mgp.p"
yumi_bnn_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_bnn.p"
yumi_mgp_result_file = "/home/shahbaz/Research/Software/model_learning/Results/results_yumi_mgp.p"

blocks_bnn_results = pickle.load( open(blocks_bnn_result_file, "rb" ))
blocks_mgp_results = pickle.load( open(blocks_mgp_result_file, "rb" ))
yumi_bnn_results = pickle.load( open(yumi_bnn_result_file, "rb" ))
yumi_mgp_results = pickle.load( open(yumi_mgp_result_file, "rb" ))

pickle.dump(blocks_bnn_results, open(blocks_bnn_result_file, "wb"), protocol=2)
pickle.dump(blocks_mgp_results, open(blocks_mgp_result_file, "wb"), protocol=2)
pickle.dump(yumi_bnn_results, open(yumi_bnn_result_file, "wb"), protocol=2)
pickle.dump(yumi_mgp_results, open(yumi_mgp_result_file, "wb"), protocol=2)

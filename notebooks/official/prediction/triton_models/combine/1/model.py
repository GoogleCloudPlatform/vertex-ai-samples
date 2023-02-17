import numpy as np
import sys
import json

import triton_python_backend_utils as pb_utils
import transformers

class TritonPythonModel:

    def initialize(self, args):
        self.log = open("/tmp/combine.loq", "w")
        self.log.write("DEBUG: ------------------------ hello world init combine/model.py------------------------------------\n")

        self.model_config = model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        self.log.write("DEBUG: ------------------------hello world execute combine/model.py\n")

        output_dtype = self.output_dtype
        responses = []
        out_tensor = []
        self.log.write("DEBUG: ------------------------requests: combine/model.py " + str(requests) + "\n")
        for request in requests:
            xgb_class = pb_utils.get_input_tensor_by_name(request, "xgb_class")
            tf_class = pb_utils.get_input_tensor_by_name(request, "tf_class")
            sci_1_class = pb_utils.get_input_tensor_by_name(request, "sci_1_class")
            sci_2_class = pb_utils.get_input_tensor_by_name(request, "sci_2_class")

            self.log.write("DEBUG: ------------------------ xgb_class tf_class sci_1_class sci_2_class \n" 
            + str(xgb_class.as_numpy())   + '\n' 
            + str(tf_class.as_numpy())    + '\n'
            + str(sci_1_class.as_numpy()) + '\n'
            + str(sci_2_class.as_numpy()) + '\n' )  

            out_tensor.append(pb_utils.Tensor("OUTPUT0", 
            (xgb_class.as_numpy() 
            + tf_class.as_numpy()  
            + sci_1_class.as_numpy() 
            + sci_2_class.as_numpy()) / 4.0))

            inference_response = pb_utils.InferenceResponse(output_tensors = out_tensor)
            responses.append(inference_response)

        self.log.flush()
        return responses

    def finalize(self):
        self.log.write("DEBUG: ------------------------ hello world finalize combine/model.py------------------------------------\n")
        self.log.write('Cleaning up - custom model combine')
        self.log.close()





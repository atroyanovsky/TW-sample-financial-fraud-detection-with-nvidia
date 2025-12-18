import boto3

def get_cfn_output(stack_name, output_key):
    try:
        cfn_client = boto3.client('cloudformation')
        response = cfn_client.describe_stacks(StackName=stack_name)
        outputs = response['Stacks'][0]['Outputs']
        for output in outputs:
            if output['OutputKey'] == output_key:
                return output['OutputValue']
        return None
    except Exception as e:
        if 'AccessDenied' in str(e) or 'UnauthorizedOperation' in str(e):
            print(f"Permission Error: You don't have permission to access CloudFormation stack '{stack_name}'.")
            print("   Please ensure the Sagemaker execution role has the 'cloudformation:DescribeStacks' permission.")
        else:
            print(f"Error accessing CloudFormation stack '{stack_name}': {str(e)}")
            print("   Please check that the stack exists and you have the permissions to read from it configured.")
        return None

def get_inference_host():
    try:
        elb_client = boto3.client('elbv2', region_name="us-east-1")
        response = elb_client.describe_load_balancers()["LoadBalancers"]
        
        for elb in response:
            if elb['LoadBalancerName'].startswith('k8s-triton'):
                return elb['DNSName']
        return None
    except Exception as e:
        if 'AccessDenied' in str(e) or 'UnauthorizedOperation' in str(e):
            print(f"Permission Error: You don't have permission to access Elastic Load Balancers.")
            print("   Please ensure the Sagemaker execution role has the 'elasticloadbalancing:DescribeLoadBalancers' permission.")
        else:
            print(f"Error accessing load balancers: {str(e)}")
            print("   Please check your AWS credentials and network connectivity.")
        return None
import * as cdk from "aws-cdk-lib";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface SageMakerNotebookRoleStackProps extends cdk.StackProps {
  /**
   * The S3 bucket ARN where the models, data and config is stored
   */
  modelBucketArn: string;
  
  /**
   * The S3 bucket ARN for the model registry
   */
  modelRegistryBucketArn: string;
}

export class SageMakerNotebookRoleStack extends cdk.Stack {
  public readonly notebookRole: iam.Role;

  constructor(
    scope: Construct,
    id: string,
    props: SageMakerNotebookRoleStackProps
  ) {
    super(scope, id, props);

    // Create the SageMaker notebook role
    this.notebookRole = new iam.Role(this, "SageMakerNotebookRole", {
      roleName: "AmazonSageMaker-NotebookRole-CDK",
      description:
        "SageMaker notebook role for training jobs, S3 access, and CloudFormation reads",
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("sagemaker.amazonaws.com")
      ),
      maxSessionDuration: cdk.Duration.hours(12),

      // AWS Managed Policies
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
      ],
    });

    // S3 access for model buckets
    const s3Policy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "S3BucketAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "s3:ListBucket",
            "s3:GetBucketLocation",
            "s3:GetBucketVersioning",
          ],
          resources: [
            props.modelBucketArn,
            props.modelRegistryBucketArn,
            "arn:aws:s3:::sagemaker-*",
          ],
        }),
        new iam.PolicyStatement({
          sid: "S3ObjectAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:AbortMultipartUpload",
            "s3:ListMultipartUploadParts",
          ],
          resources: [
            `${props.modelBucketArn}/*`,
            `${props.modelRegistryBucketArn}/*`,
            "arn:aws:s3:::sagemaker-*/*",
          ],
        }),
      ],
    });

    // CloudFormation read access
    const cfnPolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "CloudFormationReadAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "cloudformation:DescribeStacks",
            "cloudformation:DescribeStackResources",
            "cloudformation:DescribeStackResource",
            "cloudformation:GetTemplate",
            "cloudformation:ListStackResources",
            "cloudformation:ListStacks",
            "cloudformation:GetStackPolicy",
            "cloudformation:DescribeStackEvents",
          ],
          resources: [
            `arn:aws:cloudformation:${this.region}:${this.account}:stack/*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "CloudFormationListAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "cloudformation:ListStacks",
            "cloudformation:ListExports",
          ],
          resources: ["*"],
        }),
      ],
    });

    // ECR access for training images
    const ecrPolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "ECRAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "ecr:GetAuthorizationToken",
            "ecr:BatchCheckLayerAvailability",
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchGetImage",
            "ecr:DescribeRepositories",
            "ecr:DescribeImages",
            "ecr:ListImages",
          ],
          resources: ["*"],
        }),
      ],
    });

    // IAM PassRole for SageMaker training jobs
    const iamPassRolePolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "PassRoleToSageMaker",
          effect: iam.Effect.ALLOW,
          actions: ["iam:PassRole"],
          resources: [
            `arn:aws:iam::${this.account}:role/AmazonSageMaker-ExecutionRole-*`,
          ],
          conditions: {
            StringEquals: {
              "iam:PassedToService": "sagemaker.amazonaws.com",
            },
          },
        }),
      ],
    });

    // Logs access for viewing training job logs
    const logsPolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "CloudWatchLogsAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents",
            "logs:DescribeLogStreams",
            "logs:GetLogEvents",
          ],
          resources: [
            `arn:aws:logs:${this.region}:${this.account}:log-group:/aws/sagemaker/*`,
          ],
        }),
      ],
    });

    // SSM Parameter Store access for reading/writing model names
    const ssmPolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "SSMParameterAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "ssm:GetParameter",
            "ssm:GetParameters",
            "ssm:PutParameter",
          ],
          resources: [
            `arn:aws:ssm:${this.region}:${this.account}:parameter/triton/*`,
          ],
        }),
      ],
    });

    // ELB access for discovering inference endpoints
    const elbPolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "ELBDescribeAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "elasticloadbalancing:DescribeLoadBalancers",
            "elasticloadbalancing:DescribeTargetGroups",
            "elasticloadbalancing:DescribeListeners",
            "elasticloadbalancing:DescribeTags",
          ],
          resources: ["*"],
        }),
      ],
    });

    // Attach inline policies to the role
    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookS3Policy", {
        policyName: "SageMaker-Notebook-S3-Access",
        document: s3Policy,
      })
    );

    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookCFNPolicy", {
        policyName: "SageMaker-Notebook-CFN-Read",
        document: cfnPolicy,
      })
    );

    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookECRPolicy", {
        policyName: "SageMaker-Notebook-ECR-Access",
        document: ecrPolicy,
      })
    );

    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookIAMPassRolePolicy", {
        policyName: "SageMaker-Notebook-IAM-PassRole",
        document: iamPassRolePolicy,
      })
    );

    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookLogsPolicy", {
        policyName: "SageMaker-Notebook-Logs-Access",
        document: logsPolicy,
      })
    );

    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookSSMPolicy", {
        policyName: "SageMaker-Notebook-SSM-Access",
        document: ssmPolicy,
      })
    );

    this.notebookRole.attachInlinePolicy(
      new iam.Policy(this, "NotebookELBPolicy", {
        policyName: "SageMaker-Notebook-ELB-Read",
        document: elbPolicy,
      })
    );

    // Output the role ARN
    new cdk.CfnOutput(this, "SageMakerNotebookRoleArn", {
      value: this.notebookRole.roleArn,
      description: "ARN of the SageMaker notebook role",
      exportName: "SageMakerNotebookRoleArn",
    });

    new cdk.CfnOutput(this, "SageMakerNotebookRoleName", {
      value: this.notebookRole.roleName,
      description: "Name of the SageMaker notebook role",
      exportName: "SageMakerNotebookRoleName",
    });
  }
}

import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as iam from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export interface TritonImageRepoStackProps extends cdk.StackProps {}

export class TritonImageRepoStack extends cdk.Stack {
  public readonly repository: ecr.Repository;
  public readonly repositoryUri: string;

  constructor(scope: Construct, id: string, props?: TritonImageRepoStackProps) {
    super(scope, id, props);

    // ECR Repository for custom Triton image
    this.repository = new ecr.Repository(this, "TritonInferenceRepo", {
      repositoryName: "triton-inference-server",
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      lifecycleRules: [
        {
          maxImageCount: 5,
          description: "Keep only 5 most recent images",
        },
      ],
    });

    this.repositoryUri = this.repository.repositoryUri;

    // CodeBuild project to build the custom Triton image
    const buildProject = new codebuild.Project(this, "TritonImageBuild", {
      projectName: "triton-inference-image-build",
      description: "Builds custom Triton image with PyTorch, PyG, XGBoost, Captum",
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        privileged: true, // Required for Docker builds
        computeType: codebuild.ComputeType.LARGE,
      },
      environmentVariables: {
        ECR_REPO_URI: { value: this.repository.repositoryUri },
        AWS_ACCOUNT_ID: { value: this.account },
        AWS_REGION: { value: this.region },
      },
      buildSpec: codebuild.BuildSpec.fromObject({
        version: "0.2",
        phases: {
          pre_build: {
            commands: [
              "echo Logging in to Amazon ECR...",
              "aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com",
            ],
          },
          build: {
            commands: [
              "echo Building Triton image...",
              "cd triton",
              "docker build -t $ECR_REPO_URI:latest -t $ECR_REPO_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION .",
            ],
          },
          post_build: {
            commands: [
              "echo Pushing to ECR...",
              "docker push $ECR_REPO_URI:latest",
              "docker push $ECR_REPO_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION",
              "echo Build completed on `date`",
            ],
          },
        },
      }),
      timeout: cdk.Duration.hours(2),
    });

    // Grant CodeBuild permission to push to ECR
    this.repository.grantPullPush(buildProject);

    // Grant CodeBuild permission to login to ECR
    buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      })
    );

    // Outputs
    new cdk.CfnOutput(this, "TritonImageRepo", {
      value: this.repository.repositoryName,
      exportName: "TritonImageRepoName",
    });

    new cdk.CfnOutput(this, "TritonImageRepoUri", {
      value: this.repository.repositoryUri,
      exportName: "TritonImageRepoUri",
    });

    new cdk.CfnOutput(this, "TritonImageLatest", {
      value: `${this.repository.repositoryUri}:latest`,
      exportName: "TritonImageLatest",
    });

    new cdk.CfnOutput(this, "TritonBuildProject", {
      value: buildProject.projectName,
    });
  }
}

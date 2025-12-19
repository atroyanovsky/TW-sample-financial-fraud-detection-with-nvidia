import * as cdk from "aws-cdk-lib";
import { AwsSolutionsChecks } from "cdk-nag";
import { NvidiaFraudDetectionBlueprint } from "../lib/nvidia-fraud-detection-blueprint";
import { TarExtractorStack } from "../lib/tar-extractor-stack";
import { SageMakerExecutionRoleStack } from "../lib/sagemaker-training-role";
import { SageMakerNotebookRoleStack } from "../lib/sagemaker-notebook-role";
import { BlueprintECRStack } from "../lib/training-image-repo";
import { TritonImageRepoStack } from "../lib/triton-image-repo";

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION,
};

const modelBucketName = "ml-on-containers-" + process.env.CDK_DEFAULT_ACCOUNT;
const kfBucketName = "kubeflow-pipelines-" + process.env.CDK_DEFAULT_ACCOUNT;
const dataBucketName = modelBucketName;
const modelRegistryBucketName = modelBucketName + "-model-registry";

const tarExtractorStack = new TarExtractorStack(
  app,
  "NvidiaFraudDetectionBlueprintModelExtractor",
  {
    env: env,
    modelBucketName: modelBucketName,
  },
);

const sagemakerExecutionRole = new SageMakerExecutionRoleStack(
  app,
  "NvidiaFraudDetectionTrainingRole",
  {
    env: env,
    modelBucketArn: "arn:aws:s3:::" + modelBucketName,
  },
);

const sagemakerNotebookRole = new SageMakerNotebookRoleStack(
  app,
  "NvidiaFraudDetectionNotebookRole",
  {
    env: env,
    modelBucketArn: "arn:aws:s3:::" + modelBucketName,
    modelRegistryBucketArn: "arn:aws:s3:::" + modelRegistryBucketName,
  },
);

const trainingImageRepo = new BlueprintECRStack(
  app,
  "NvidiaFraudDetectionTrainingImageRepo",
  {
    env: env,
  },
);

const tritonImageRepo = new TritonImageRepoStack(
  app,
  "NvidiaFraudDetectionTritonImageRepo",
  { env: env },
);

const mainStack = new NvidiaFraudDetectionBlueprint(
  app,
  "NvidiaFraudDetectionBlueprint",
  {
    env: env,
    modelBucketName: modelRegistryBucketName,
    kubeflowBucketName: kfBucketName,
    dataBucketName: dataBucketName,
    modelRegistryBucketName: modelRegistryBucketName,
    tritonImageUri: `${tritonImageRepo.repositoryUri}:latest`,
  },
);

mainStack.addDependency(trainingImageRepo);
mainStack.addDependency(tritonImageRepo);
mainStack.addDependency(sagemakerExecutionRole);
mainStack.addDependency(sagemakerNotebookRole);
mainStack.addDependency(tarExtractorStack);

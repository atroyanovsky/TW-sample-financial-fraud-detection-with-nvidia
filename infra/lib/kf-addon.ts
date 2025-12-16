import {ArgoCDAddOn, ClusterAddOn, ClusterInfo } from "@aws-quickstart/eks-blueprints";
import {Construct} from "constructs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as eks from "aws-cdk-lib/aws-eks";
import { CfnJson} from "aws-cdk-lib";

export interface KfAddonProps {
    bucketName: string
}

export class KfAddon implements ClusterAddOn {
    constructor(readonly props: KfAddonProps) {}

    deploy(clusterInfo: ClusterInfo): Promise<Construct> | void {
        const kfPolicyDoc = iam.PolicyDocument.fromJson(KfIAMPolicy(this.props.bucketName));

        const kfPolicy = new iam.ManagedPolicy(clusterInfo.cluster, "KubeflowPolicy", { document: kfPolicyDoc});

        const conditions = new CfnJson(clusterInfo.cluster, 'ConditionJson', {
          value: {
            [`${clusterInfo.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:aud`]: 'sts.amazonaws.com',
            [`${clusterInfo.cluster.openIdConnectProvider.openIdConnectProviderIssuer}:sub`]: `system:serviceaccount:*`,
          },
        });
        const principal = new iam.OpenIdConnectPrincipal(clusterInfo.cluster.openIdConnectProvider).withConditions({
          StringLike: conditions,
        });
        const kfRole = new iam.Role(clusterInfo.cluster, 'kf-role', { assumedBy: principal });
        kfRole.addManagedPolicy(kfPolicy);

        clusterInfo.cluster.addManifest("deployKF-argo-app", deployKfApp(this.props.bucketName, clusterInfo.cluster.stack.region, kfRole.roleArn))
    }

}
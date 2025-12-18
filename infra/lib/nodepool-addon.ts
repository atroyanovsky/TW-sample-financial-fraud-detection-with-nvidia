import { ClusterAddOn, ClusterInfo } from "@aws-quickstart/eks-blueprints";
import { Construct } from "constructs";
import * as blueprints from "@aws-quickstart/eks-blueprints";
import { dependable } from "@aws-quickstart/eks-blueprints/dist/utils";

export interface NodePoolAddonProps {
  nodePoolSpec: blueprints.NodePoolV1Spec;
  ec2NodeClassSpec?: blueprints.Ec2NodeClassV1Spec;
}

export class NodePoolAddon implements ClusterAddOn {
  constructor(readonly props: NodePoolAddonProps) { }

  /**
   * Helper function to convert a key-pair values (with an operator)
   * of spec configurations to appropriate json format for addManifest function
   * @param reqs
   * @returns newReqs
   * */
  protected convert(reqs: { key: string; operator: string; values: string[] }[]): any[] {
    const newReqs = [];
    for (let req of reqs) {
      const key = req["key"];
      const op = req["operator"];
      const val = req["values"];
      const requirement = { key: key, operator: op, values: val };
      newReqs.push(requirement);
    }
    return newReqs;
  }

  @dependable(blueprints.addons.KarpenterV1AddOn.name)
  deploy(clusterInfo: ClusterInfo): Promise<Construct> {
    const nodePool = clusterInfo.cluster.addManifest("additional-nodepool", {
      apiVersion: "karpenter.sh/v1",
      kind: "NodePool",
      metadata: {
        name: "additional-nodepool"
      },
      spec: {
        template: {
          metadata: { labels: this.props.nodePoolSpec.labels, annotations: this.props.nodePoolSpec.annotations },
          spec: {
            nodeClassRef: {
              name: "default-ec2nodeclass",
              group: "karpenter.k8s.aws",
              kind: "EC2NodeClass"
            },
            taints: this.props.nodePoolSpec.taints,
            startupTaints: this.props.nodePoolSpec.startupTaints,
            requirements: this.convert(this.props.nodePoolSpec.requirements || []),
            expireAfter: this.props.nodePoolSpec.expireAfter
          },
        },
        disruption: this.props.nodePoolSpec.disruption,
        limits: this.props.nodePoolSpec.limits,
        weight: this.props.nodePoolSpec.weight,
      }
    });

    if (this.props.ec2NodeClassSpec) {
      const ec2NodeClass = clusterInfo.cluster.addManifest("additional-ec2nodeclass", {
        apiVersion: "karpenter.k8s.aws/v1",
        kind: "EC2NodeClass",
        metadata: {
          name: "additional-ec2nodeclass"
        },
        spec: this.props.ec2NodeClassSpec
      });

      nodePool.node.addDependency(ec2NodeClass);
    }

    return Promise.resolve(nodePool);
  }
}

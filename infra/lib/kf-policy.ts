export const KfIAMPolicy = (bucket: string, dataBucket: string, modelBucket: string) => {
  return {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetBucketLocation",
          "s3:ListBucket"
        ],
        "Resource": [
          `arn:aws:s3:::${bucket}`,
          `arn:aws:s3:::${dataBucket}`,
          `arn:aws:s3:::${modelBucket}`
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ],
        "Resource": [
          `arn:aws:s3:::${bucket}/artifacts/*`,
          `arn:aws:s3:::${bucket}/pipelines/*`,
          `arn:aws:s3:::${bucket}/v2/artifacts/*`
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject"
        ],
        "Resource": [
          `arn:aws:s3:::${dataBucket}/*`
        ]
      },
      {
        "Effect": "Allow",
        "Action": [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ],
        "Resource": [
          `arn:aws:s3:::${modelBucket}/*`
        ]
      }
    ]
  }
}

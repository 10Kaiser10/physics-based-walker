using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    //for genetic algo
    public float score;

    //body parts
    public Rigidbody torso;
    public Rigidbody pelvis;
    public Rigidbody thighL;
    public Rigidbody thighR;
    public Rigidbody legL;
    public Rigidbody legR;
    public Rigidbody feetL;
    public Rigidbody feetR;
    private Rigidbody[] rbArr;
    private HingeJoint[] hJoints;
    private BodyPartCollision[] bpcArr;

    //neural network
    private int inputNodes = 58;
    //private int outputNodes = 24;
    private int[] hiddenLayersNodes = { 50, 40, 32 };
    private int[] LayersNodes = { 58, 50, 40, 32, 24 };
    private float[][,] weightsNBiases;

    //feet touching check
    public BodyPartCollision Lfoot;
    public BodyPartCollision Rfoot;

    void Start()
    {
        rbArr = new Rigidbody[] { torso, pelvis, thighL, thighR, legL, legR, feetL, feetR };
        hJoints = new HingeJoint[rbArr.Length];
        bpcArr = new BodyPartCollision[rbArr.Length];
        for(int i=0; i<rbArr.Length; i++)
        {
            hJoints[i] = rbArr[i].gameObject.GetComponent<HingeJoint>();
            bpcArr[i] = rbArr[i].gameObject.GetComponent<BodyPartCollision>();
        }

        weightsNBiases = new float[hiddenLayersNodes.Length + 1][,];

        for(int i=0; i<=hiddenLayersNodes.Length; i++)
        {
            int rows = LayersNodes[i + 1];
            int cols = LayersNodes[i] + 1;
            weightsNBiases[i] = new float[rows, cols];

            for(int j=0; j<rows; j++)
            {
                for(int k=0; k<cols; k++)
                {
                    weightsNBiases[i][j, k] = Random.Range(-1000, 1000);
                }
            }
        }

        score = 0;
    }

    float leakyRelu(float input)
    {
        if(input>0)
        {
            return input;
        }
        else
        {
            return 0.01f * input;
        }
    }

    float[] leakyRelu(float[] input)
    {
        float[] output = new float[input.Length + 1];
        output[0] = 1;

        for(int i=1; i<=input.Length; i++)
        {
            output[i] = leakyRelu(input[i - 1]);
        }
        
        return output;
    }

    float[] matrixMult(float[,] matrix, float[] vector)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        float[] output = new float[rows];

        for(int i=0;i<rows;i++)
        {
            output[i] = 0;
            for(int j=0;j<cols;j++)
            {
                output[i] += matrix[i,j] * vector[j];
            }
        }

        return output;
    }

    float[] NeuralNetwork(float[] input)
    {
        float[] currentVector;
        currentVector = new float[input.Length + 1];
        currentVector[0] = 1;

        for (int i = 1; i <= input.Length; i++)
        {
            currentVector[i] = input[i - 1];
        }

        for (int i=0; i<hiddenLayersNodes.Length; i++)
        {
            currentVector = matrixMult(weightsNBiases[i], currentVector);
            currentVector = leakyRelu(currentVector);
        }

        return matrixMult(weightsNBiases[hiddenLayersNodes.Length], currentVector);
    }

    void move()
    {
        float[] inputVec = new float[inputNodes];

        for (int i = 0; i < 8; i++)
        {
            Vector3 pos = rbArr[i].transform.localPosition;
            inputVec[i * 3] = pos.x;
            inputVec[i * 3 + 1] = pos.y;
            inputVec[i * 3 + 2] = pos.z;
        }

        for (int i = 0; i < 8; i++)
        {
            Vector3 vel = rbArr[i].velocity;
            inputVec[24 + i * 3] = vel.x;
            inputVec[24 + i * 3 + 1] = vel.y;
            inputVec[24 + i * 3 + 2] = vel.z;
        }

        for (int i = 0; i < 4; i++)
        {
            inputVec[48 + i * 2] = hJoints[i + 2].angle;
            inputVec[48 + i * 2 + 1] = hJoints[i + 2].velocity;
        }

        inputVec[56] = Lfoot.collided;
        inputVec[57] = Rfoot.collided;
        Lfoot.collided = 0;
        Rfoot.collided = 0;

        float[] outputArr = NeuralNetwork(inputVec);

        Vector3[] torques = new Vector3[8];
        for (int i = 0; i < 8; i++)
        {
            torques[i] = new Vector3(outputArr[3 * i], outputArr[3 * i + 1], outputArr[3 * i + 2]);
        }

        Vector3 mean = Vector3.zero;
        for (int i = 0; i < 8; i++)
        {
            mean += torques[i];
        }
        mean = mean / 8;
        for (int i = 0; i < 8; i++)
        {
            torques[i] -= mean;
        }

        for (int i = 0; i < 8; i++)
        {
            rbArr[i].AddRelativeTorque(torques[i]);
        }
    }

    void calcScore()
    {
        float touches = 0;
        for(int i=0; i<6; i++)
        {
            touches += bpcArr[i].collided;
            bpcArr[i].collided = 0;
        }

        score -= touches / 100;
    }

    void FixedUpdate()
    {
        move();
        calcScore();
    }
}

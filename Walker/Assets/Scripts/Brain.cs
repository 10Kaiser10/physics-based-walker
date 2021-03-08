using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    //for genetic algo
    public float score;

    //body parts
    public GameObject[] bodyParts;
    private ArticulationBody[] artBodies;
    private BodyPartCollision[] bpcArr;
    private ArticulationDrive[] artDrives;

    //neural network
    [System.NonSerialized]
    public int inputNodes = 76;
    [System.NonSerialized]
    public int outputNodes = 13;
    [System.NonSerialized]
    public int[] hiddenLayersNodes = { 56, 36 };
    [System.NonSerialized]
    public int[] LayersNodes = { 76, 56, 36, 13 };
    [System.NonSerialized]
    public float[][,] weightsNBiases;

    void Start()
    {
        artBodies = new ArticulationBody[bodyParts.Length];
        bpcArr = new BodyPartCollision[bodyParts.Length];
        for (int i = 0; i < bodyParts.Length; i++)
        {
            artBodies[i] = bodyParts[i].GetComponent<ArticulationBody>();
            bpcArr[i] = bodyParts[i].transform.GetChild(0).GetComponent<BodyPartCollision>();
        }

        score = 0;
    }

    float putInBetween(float l, float h, float n)
    {
        float diff = h - l;
        float off = n % diff;
        if (off < 0) { off += diff; }
        return l + off;
    }

    float leakyRelu(float input)
    {
        if (input > 0)
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

        for (int i = 1; i <= input.Length; i++)
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

        for (int i = 0; i < rows; i++)
        {
            output[i] = 0;
            for (int j = 0; j < cols; j++)
            {
                output[i] += matrix[i, j] * vector[j];
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

        for (int i = 0; i < hiddenLayersNodes.Length; i++)
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
            Vector3 pos = bodyParts[i].transform.localPosition;
            inputVec[i * 3] = pos.x;
            inputVec[i * 3 + 1] = pos.y;
            inputVec[i * 3 + 2] = pos.z;
        }

        for (int i = 0; i < 8; i++)
        {
            Vector3 vel = artBodies[i].velocity;
            inputVec[24 + i * 3] = vel.x;
            inputVec[24 + i * 3 + 1] = vel.y;
            inputVec[24 + i * 3 + 2] = vel.z;
        }

        int[] index = new int[3] { 0, 2, 3 };
        for(int i = 0; i<3; i++)
        {
            inputVec[48 + 0 + i * 6] = artBodies[index[i]].jointVelocity[0];
            inputVec[48 + 1 + i * 6] = artBodies[index[i]].jointVelocity[1];
            inputVec[48 + 2 + i * 6] = artBodies[index[i]].jointVelocity[2];
            inputVec[48 + 3 + i * 6] = artBodies[index[i]].jointPosition[0];
            inputVec[48 + 4 + i * 6] = artBodies[index[i]].jointPosition[1];
            inputVec[48 + 5 + i * 6] = artBodies[index[i]].jointPosition[2];
        }

        index = new int[4] { 4, 5, 6, 7};
        for(int i=0; i<4; i++)
        {
            inputVec[66 + 0 + i * 2] = artBodies[index[i]].jointVelocity[0];
            inputVec[66 + 1 + i * 2] = artBodies[index[i]].jointPosition[0];
        }

        inputVec[74] = bpcArr[6].collided;
        inputVec[75] = bpcArr[7].collided;
        bpcArr[6].collided = 0;
        bpcArr[7].collided = 0;

        float[] outputArr = NeuralNetwork(inputVec);

        artDrives = new ArticulationDrive[13];

        index = new int[3] { 0, 2, 3 };
        for (int i = 0; i < 3; i++)
        {
            artDrives[3 * i + 0] = artBodies[index[i]].xDrive;
            artDrives[3 * i + 1] = artBodies[index[i]].yDrive;
            artDrives[3 * i + 2] = artBodies[index[i]].zDrive;
        }

        index = new int[4] { 4, 5, 6, 7 };
        for (int i = 0; i < 4; i++)
        {
            artDrives[9 + i] = artBodies[index[i]].xDrive;
        }

        for(int i=0; i<13; i++)
        {
            artDrives[i].target = putInBetween(artDrives[i].lowerLimit, artDrives[i].upperLimit, outputArr[i]);
        }

        index = new int[3] { 0, 2, 3 };
        for (int i = 0; i < 3; i++)
        {
            artBodies[index[i]].xDrive = artDrives[3 * i + 0];
            artBodies[index[i]].yDrive = artDrives[3 * i + 1];
            artBodies[index[i]].zDrive = artDrives[3 * i + 2];
        }

        index = new int[4] { 4, 5, 6, 7 };
        for (int i = 0; i < 4; i++)
        {
            artBodies[index[i]].xDrive = artDrives[9 + i];
        }
    }

    void calcScore()
    {
        float touches = 0;
        for (int i = 0; i < 6; i++)
        {
            touches += bpcArr[i].collided;
            bpcArr[i].collided = 0;
        }

        //score -= touches / 100;
        score += (bodyParts[0].transform.position.y);
    }

    void FixedUpdate()
    {
        move();
        calcScore();
    }
}

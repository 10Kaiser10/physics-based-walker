using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Evolver : MonoBehaviour
{
    //timer
    private float timer;
    public float simDuration = 10;

    //evolution
    public float groundTouchWieght = 1;
    public float distanceWieght = 1;

    //population
    public int populationSize = 16;
    private GameObject[] population;
    private float[] scores;

    //spawning
    public int spawnRows = 20;
    public GameObject prefab;
    public GameObject parent;

    //neural network
    private int inputNodes = 58;
    private int outputNodes = 24;
    private int[] hiddenLayersNodes = { 50, 40, 32 };
    private int[] LayersNodes = { 58, 50, 40, 32, 24 };
    private float[][][,] weightsNBiases;
    private float[][][,] nextWeightsNBiases;

    public void Begin()
    {
        scores = new float[populationSize];
        weightsNBiases = new float[populationSize][][,];
        
        for(int i=0; i<populationSize; i++)
        {
            weightsNBiases[i] = new float[hiddenLayersNodes.Length + 1][,];

            for (int l = 0; l <= hiddenLayersNodes.Length; l++)
            {
                int rows = LayersNodes[l + 1];
                int cols = LayersNodes[l] + 1;
                weightsNBiases[i][l] = new float[rows, cols];

                for (int j = 0; j < rows; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        weightsNBiases[i][l][j, k] = Random.Range(-1000, 1000);
                    }
                }
            }
        }

        population = new GameObject[populationSize];

        for(int i=0; i<populationSize; i++)
        {
            population[i] = Instantiate(prefab, new Vector3(10*(i/spawnRows), 0.983f, 10*(i%spawnRows)), Quaternion.identity, parent.transform);
            Brain objBrain = population[i].transform.GetChild(0).GetComponent<Brain>();
            objBrain.inputNodes = inputNodes;
            objBrain.outputNodes = outputNodes;
            objBrain.hiddenLayersNodes = hiddenLayersNodes;
            objBrain.LayersNodes = LayersNodes;
            objBrain.weightsNBiases = weightsNBiases[i];
        }
    }

    private void Start()
    {
        timer = 0;
        Begin();
    }

    private void calcScores()
    {
        for(int i=0;i<populationSize; i++)
        {
            scores[i] = groundTouchWieght*population[i].transform.GetChild(0).GetComponent<Brain>().score;
            scores[i] += -distanceWieght*(population[i].transform.GetChild(2).transform.position.z - population[i].transform.position.z);
        }
    }

    private void selection()
    {
        float scoreSum = 0;
        int elite = 0;
        for(int i=0; i<populationSize; i++)
        {
            scoreSum += scores[i];
            if (scores[elite] < scores[i]) { elite = i; }
        }
        Debug.Log(scoreSum / populationSize);


        nextWeightsNBiases = new float[populationSize][][,];

        nextWeightsNBiases[0] = new float[hiddenLayersNodes.Length + 1][,];

        for (int l = 0; l <= hiddenLayersNodes.Length; l++)
        {
            int rows = LayersNodes[l + 1];
            int cols = LayersNodes[l] + 1;
            nextWeightsNBiases[0][l] = new float[rows, cols];

            for (int j = 0; j < rows; j++)
            {
                for (int k = 0; k < cols; k++)
                {
                    nextWeightsNBiases[0][l][j, k] = weightsNBiases[elite][l][j, k];
                }
            }
        }
    }

    private void crossover()
    {
        int[] parents = new int[populationSize - 1];
        int comp1, comp2;
        for(int i=0; i<populationSize-1; i++)
        {
            comp1 = Random.Range(0, populationSize);
            do
            {
                comp2 = Random.Range(0, populationSize);
            } while (comp1 == comp2);

            if(scores[comp1] > scores[comp2]) { parents[i] = comp1; }
            else { parents[i] = comp2; }
        }

        for (int i = 1; i < populationSize; i++)
        {
            nextWeightsNBiases[i] = new float[hiddenLayersNodes.Length + 1][,];
            int par1 = parents[Random.Range(0, populationSize - 1)], par2 = parents[Random.Range(0, populationSize - 1)];

            for (int l = 0; l <= hiddenLayersNodes.Length; l++)
            {
                int rows = LayersNodes[l + 1];
                int cols = LayersNodes[l] + 1;
                nextWeightsNBiases[i][l] = new float[rows, cols];

                for (int j = 0; j < rows; j++)
                {
                    for (int k = 0; k < cols; k++)
                    {
                        float coeff = Random.Range(0.01f, 0.99f);
                        nextWeightsNBiases[i][l][j, k] = coeff*weightsNBiases[par1][l][j, k] + (1-coeff)*weightsNBiases[par2][l][j, k];
                    }
                }
            }
        }
    }

    private void respawn()
    {
        foreach(GameObject obj in population)
        {
            Destroy(obj);
        }

        population = new GameObject[populationSize];

        for (int i = 0; i < populationSize; i++)
        {
            population[i] = Instantiate(prefab, new Vector3(10 * (i / spawnRows), 0.983f, 10 * (i % spawnRows)), Quaternion.identity, parent.transform);
            Brain objBrain = population[i].transform.GetChild(0).GetComponent<Brain>();
            objBrain.inputNodes = inputNodes;
            objBrain.outputNodes = outputNodes;
            objBrain.hiddenLayersNodes = hiddenLayersNodes;
            objBrain.LayersNodes = LayersNodes;
            objBrain.weightsNBiases = weightsNBiases[i];
        }
    }

    private void Update()
    {
        timer += Time.deltaTime;

        if(timer > simDuration)
        {
            timer = 0;
            calcScores();
            selection();
            crossover();

            weightsNBiases = nextWeightsNBiases;

            respawn();
        }
    }

}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BodyPartCollision : MonoBehaviour
{
    public float collided;
    void Start()
    {
        collided = 0;
    }

    private void OnCollisionStay(Collision collision)
    {
        collided = 1;
    }
}

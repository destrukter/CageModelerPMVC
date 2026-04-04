using UnityEngine;

/// <summary>
/// Enables a projection plane and swaps between three textures based on which object is inside this trigger.
///
/// Conditions required to show the projection:
/// 1) One of the configured trigger objects is inside this trigger.
/// 2) Enough light-off events were registered via RegisterLightTurnedOffEvent().
/// 3) Door state was set to closed via SetDoorClosedState(true).
///
/// Hook RegisterLightTurnedOffEvent to your LightTurnsOffWhenPushedOver event.
/// Hook SetDoorClosedState(bool) to your DoorRandomYTorqueNotifier closed/open signal.
/// </summary>
public class ProjectionAssetTriggerController : MonoBehaviour
{
    [Header("Projection")]
    [SerializeField] private GameObject projectionPlane;
    [SerializeField] private Renderer projectionRenderer;

    [Header("Objects -> Projection Texture")]
    [SerializeField] private string object1Tag = "ProjectionTargetA";
    [SerializeField] private Texture object1Texture;
    [SerializeField] private string object2Tag = "ProjectionTargetB";
    [SerializeField] private Texture object2Texture;
    [SerializeField] private string object3Tag = "ProjectionTargetC";
    [SerializeField] private Texture object3Texture;

    [Header("Gate Conditions")]
    [SerializeField] private int requiredLightOffEvents = 1;

    private int lightOffEventCount;
    private bool isDoorClosed;
    private Collider currentProjectionTarget;

    private void Awake()
    {
        if (projectionPlane == null)
        {
            Debug.LogError($"{nameof(ProjectionAssetTriggerController)} on {name} needs a projection plane reference.");
            enabled = false;
            return;
        }

        if (projectionRenderer == null)
        {
            projectionRenderer = projectionPlane.GetComponent<Renderer>();
        }

        projectionPlane.SetActive(false);
    }

    private void OnTriggerEnter(Collider other)
    {
        if (!IsSupportedProjectionObject(other, out _))
        {
            return;
        }

        currentProjectionTarget = other;
        EvaluateProjectionState();
    }

    private void OnTriggerExit(Collider other)
    {
        if (currentProjectionTarget != other)
        {
            return;
        }

        currentProjectionTarget = null;
        EvaluateProjectionState();
    }

    /// <summary>
    /// Call this from LightTurnsOffWhenPushedOver event each time one light-off event occurs.
    /// </summary>
    public void RegisterLightTurnedOffEvent()
    {
        lightOffEventCount++;
        EvaluateProjectionState();
    }

    /// <summary>
    /// Call this from DoorRandomYTorqueNotifier with true when door is closed, false when open.
    /// </summary>
    public void SetDoorClosedState(bool closed)
    {
        isDoorClosed = closed;
        EvaluateProjectionState();
    }

    private void EvaluateProjectionState()
    {
        if (currentProjectionTarget == null)
        {
            projectionPlane.SetActive(false);
            return;
        }

        bool hasEnoughLightEvents = lightOffEventCount >= requiredLightOffEvents;
        if (!hasEnoughLightEvents || !isDoorClosed)
        {
            projectionPlane.SetActive(false);
            return;
        }

        if (!IsSupportedProjectionObject(currentProjectionTarget, out Texture textureToUse) || textureToUse == null)
        {
            projectionPlane.SetActive(false);
            return;
        }

        if (projectionRenderer != null)
        {
            projectionRenderer.material.mainTexture = textureToUse;
        }

        projectionPlane.SetActive(true);
    }

    private bool IsSupportedProjectionObject(Collider other, out Texture texture)
    {
        if (other.CompareTag(object1Tag))
        {
            texture = object1Texture;
            return true;
        }

        if (other.CompareTag(object2Tag))
        {
            texture = object2Texture;
            return true;
        }

        if (other.CompareTag(object3Tag))
        {
            texture = object3Texture;
            return true;
        }

        texture = null;
        return false;
    }
}

package com.example.finger_drawing

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.airbnb.lottie.compose.*
import com.example.finger_drawing.ui.theme.FingerdrawingTheme
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity(), HandLandmarkerHelper.LandmarkerListener {

    internal lateinit var backgroundExecutor: ExecutorService
    internal lateinit var handLandmarkerHelper: HandLandmarkerHelper
    internal var isHandLandmarkerInitialized = false

    // Shared states for fingertip position and input image size
    private val _fingerTipPosition = mutableStateOf<Offset?>(null)
    val fingerTipPosition: State<Offset?> get() = _fingerTipPosition

    // Drawing path state - store multiple separate paths
    private val _drawingPaths = mutableStateOf<List<List<Offset>>>(emptyList())
    val drawingPaths: State<List<List<Offset>>> get() = _drawingPaths

    // Current path being drawn
    private val _currentPath = mutableStateOf<List<Offset>>(emptyList())
    val currentPath: State<List<Offset>> get() = _currentPath

    // Track drawing state to detect when to start new paths
    private val _isCurrentlyDrawing = mutableStateOf(false)
    val isCurrentlyDrawing: State<Boolean> get() = _isCurrentlyDrawing

    // Hand gesture state
    private val _isHandOpen = mutableStateOf(false)
    val isHandOpen: State<Boolean> get() = _isHandOpen

    private val _isLeftHand = mutableStateOf(false)
    val isLeftHand: State<Boolean> get() = _isLeftHand

    // New state for horizontal open palm detection
    private val _isHorizontalOpenPalm = mutableStateOf(false)
    val isHorizontalOpenPalm: State<Boolean> get() = _isHorizontalOpenPalm

    private val _palmCenter = mutableStateOf<Offset?>(null)
    val palmCenter: State<Offset?> get() = _palmCenter

    private val _palmDirection = mutableStateOf<Pair<Float, Float>?>(null)
    val palmDirection: State<Pair<Float, Float>?> get() = _palmDirection

    // States for both hands horizontal open palm detection
    private val _leftHandHorizontalOpen = mutableStateOf(false)
    val leftHandHorizontalOpen: State<Boolean> get() = _leftHandHorizontalOpen

    private val _rightHandHorizontalOpen = mutableStateOf(false)
    val rightHandHorizontalOpen: State<Boolean> get() = _rightHandHorizontalOpen

    private val _leftPalmCenter = mutableStateOf<Offset?>(null)
    val leftPalmCenter: State<Offset?> get() = _leftPalmCenter

    private val _rightPalmCenter = mutableStateOf<Offset?>(null)
    val rightPalmCenter: State<Offset?> get() = _rightPalmCenter

    private val _leftPalmDirection = mutableStateOf<Pair<Float, Float>?>(null)
    val leftPalmDirection: State<Pair<Float, Float>?> get() = _leftPalmDirection

    private val _rightPalmDirection = mutableStateOf<Pair<Float, Float>?>(null)
    val rightPalmDirection: State<Pair<Float, Float>?> get() = _rightPalmDirection

    private val _previewSize = mutableStateOf<Pair<Int, Int>?>(null)
    val previewSize: State<Pair<Int, Int>?> get() = _previewSize

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize background executor
        backgroundExecutor = Executors.newSingleThreadExecutor()

        enableEdgeToEdge()
        setContent {
            FingerdrawingTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    // Pass fingerTipPosition and previewSize as state into the camera screen
                    CameraScreen(
                        fingerTipPosition = fingerTipPosition,
                        drawingPaths = drawingPaths,
                        isHandOpen = isHandOpen,
                        isLeftHand = isLeftHand,
                        isHorizontalOpenPalm = isHorizontalOpenPalm,
                        palmCenter = palmCenter,
                        palmDirection = palmDirection,
                        leftHandHorizontalOpen = leftHandHorizontalOpen,
                        rightHandHorizontalOpen = rightHandHorizontalOpen,
                        leftPalmCenter = leftPalmCenter,
                        rightPalmCenter = rightPalmCenter,
                        leftPalmDirection = leftPalmDirection,
                        rightPalmDirection = rightPalmDirection,
                        previewSize = previewSize,
                        isCurrentlyDrawing = isCurrentlyDrawing,
                        currentPath = currentPath
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        backgroundExecutor.shutdown()
        if (isHandLandmarkerInitialized) {
            handLandmarkerHelper.clearHandLandmarker()
        }
    }

    override fun onError(error: String, errorCode: Int) {
        Log.e("HandTracking", "Hand landmarker error: $error")
    }

    override fun onResults(resultBundle: HandLandmarkerHelper.ResultBundle) {
        val results = resultBundle.results
        if (results.isNotEmpty()) {
            val handResult = results.first()
            if (handResult.landmarks().isNotEmpty() && handResult.handedness().isNotEmpty()) {

                // Process all detected hands to find both left and right
                var leftHandOpen: Boolean? = null
                var rightHandPosition: Offset? = null
                var foundLeftHand = false
                var foundRightHand = false

                for (handIndex in handResult.landmarks().indices) {
                    if (handIndex < handResult.handedness().size) {
                        val landmarks = handResult.landmarks()[handIndex]
                        val handedness = handResult.handedness()[handIndex]

                        // Get index finger tip (landmark 8)
                        if (landmarks.size > 8) {
                            val fingerTip = landmarks[8]
                            val wrist = landmarks[0]

                            // Convert normalized coordinates to pixel coordinates
                            val fingerTipX = (fingerTip.x() * resultBundle.inputImageWidth)
                            val fingerTipY = (fingerTip.y() * resultBundle.inputImageHeight)
                            val wristX = (wrist.x() * resultBundle.inputImageWidth)
                            val wristY = (wrist.y() * resultBundle.inputImageHeight)

                            // Use MediaPipe's handedness detection
                            val handenessLabel = handedness.first().categoryName()
                            val isLeft = handenessLabel == "Left"

                            // Check if hand is open
                            val isOpen = if (isLeft) {
                                checkIfIndexFingerOnly(
                                    landmarks,
                                    resultBundle.inputImageWidth,
                                    resultBundle.inputImageHeight
                                )
                            } else {
                                checkIfHandIsOpen(
                                    landmarks,
                                    resultBundle.inputImageWidth,
                                    resultBundle.inputImageHeight
                                )
                            }

                            // Check if hand is horizontal open palm
                            val isHorizontalOpen = checkIfHandIsHorizontalOpenPalm(
                                landmarks,
                                resultBundle.inputImageWidth,
                                resultBundle.inputImageHeight
                            )

                            Log.d(
                                "HandTracking",
                                "Hand $handIndex - Finger tip: (${fingerTipX.toInt()}, ${fingerTipY.toInt()}), " +
                                        "Handedness: $handenessLabel, Left: $isLeft, Open: $isOpen, Horizontal Open: $isHorizontalOpen, " +
                                        "MediaPipe says: $handenessLabel (Left=$isLeft)"
                            )

                            if (isLeft) {
                                // MediaPipe "Left" = actual RIGHT hand (due to mirroring)
                                // Right hand provides drawing position
                                rightHandPosition = Offset(fingerTipX, fingerTipY)
                                foundRightHand = true
                            } else {
                                // MediaPipe "Right" = actual LEFT hand (due to mirroring)
                                // Left hand controls drawing on/off
                                leftHandOpen = isOpen
                                foundLeftHand = true
                            }

                            if (isHorizontalOpen) {
                                if (isLeft) {
                                    _leftHandHorizontalOpen.value = true
                                    // Calculate palm center as average of key palm landmarks
                                    val palmLandmarks =
                                        listOf(0, 5, 9, 13, 17) // wrist, base of each finger
                                    var centerX = 0f
                                    var centerY = 0f
                                    for (landmarkIndex in palmLandmarks) {
                                        centerX += landmarks[landmarkIndex].x() * resultBundle.inputImageWidth
                                        centerY += landmarks[landmarkIndex].y() * resultBundle.inputImageHeight
                                    }
                                    centerX /= palmLandmarks.size
                                    centerY /= palmLandmarks.size
                                    _leftPalmCenter.value = Offset(centerX, centerY)

                                    val palmDirection = getPalmDirection(
                                        landmarks,
                                        resultBundle.inputImageWidth,
                                        resultBundle.inputImageHeight
                                    )
                                    _leftPalmDirection.value = palmDirection

                                    Log.d(
                                        "HandTracking",
                                        "LEFT HAND horizontal palm detected at ($centerX, $centerY)"
                                    )
                                } else {
                                    _rightHandHorizontalOpen.value = true
                                    // Calculate palm center as average of key palm landmarks
                                    val palmLandmarks =
                                        listOf(0, 5, 9, 13, 17) // wrist, base of each finger
                                    var centerX = 0f
                                    var centerY = 0f
                                    for (landmarkIndex in palmLandmarks) {
                                        centerX += landmarks[landmarkIndex].x() * resultBundle.inputImageWidth
                                        centerY += landmarks[landmarkIndex].y() * resultBundle.inputImageHeight
                                    }
                                    centerX /= palmLandmarks.size
                                    centerY /= palmLandmarks.size
                                    _rightPalmCenter.value = Offset(centerX, centerY)

                                    val palmDirection = getPalmDirection(
                                        landmarks,
                                        resultBundle.inputImageWidth,
                                        resultBundle.inputImageHeight
                                    )
                                    _rightPalmDirection.value = palmDirection

                                    Log.d(
                                        "HandTracking",
                                        "RIGHT HAND horizontal palm detected at ($centerX, $centerY)"
                                    )
                                }
                            } else {
                                if (isLeft) {
                                    _leftHandHorizontalOpen.value = false
                                    _leftPalmCenter.value = null
                                    _leftPalmDirection.value = null
                                    Log.d("HandTracking", "LEFT HAND horizontal palm NOT detected")
                                } else {
                                    _rightHandHorizontalOpen.value = false
                                    _rightPalmCenter.value = null
                                    _rightPalmDirection.value = null
                                    Log.d("HandTracking", "RIGHT HAND horizontal palm NOT detected")
                                }
                            }
                        }
                    }
                }

                // Only show UI and enable drawing when BOTH hands are detected
                if (foundLeftHand && foundRightHand && leftHandOpen != null && rightHandPosition != null) {
                    _fingerTipPosition.value = rightHandPosition
                    _isHandOpen.value = leftHandOpen!! // Left hand controls this
                    _isLeftHand.value = false  // We're showing right hand fingertip
                    _previewSize.value =
                        Pair(resultBundle.inputImageWidth, resultBundle.inputImageHeight)

                    Log.d(
                        "HandTracking",
                        "Both hands detected - Left hand open: $leftHandOpen, Drawing at right fingertip"
                    )

                    // Check if both hands are in horizontal open palm mode - if so, disable drawing
                    val bothHandsHorizontalOpen =
                        _leftHandHorizontalOpen.value && _rightHandHorizontalOpen.value

                    if (!bothHandsHorizontalOpen) {
                        // Add drawing points if LEFT hand is open (drawing at RIGHT hand position)
                        if (leftHandOpen) {
                            if (!_isCurrentlyDrawing.value) {
                                // Start new drawing path
                                _isCurrentlyDrawing.value = true
                                _currentPath.value = listOf(rightHandPosition!!)
                            } else {
                                // Continue current drawing path
                                _currentPath.value = _currentPath.value + rightHandPosition!!
                            }
                        } else {
                            // Hand closed - stop drawing and add current path to drawing paths
                            _isCurrentlyDrawing.value = false
                            if (_currentPath.value.isNotEmpty()) {
                                _drawingPaths.value =
                                    _drawingPaths.value + listOf(_currentPath.value)
                                _currentPath.value = emptyList()
                            }
                        }
                    } else {
                        // Both hands horizontal open - stop any current drawing
                        if (_isCurrentlyDrawing.value && _currentPath.value.isNotEmpty()) {
                            _drawingPaths.value = _drawingPaths.value + listOf(_currentPath.value)
                            _currentPath.value = emptyList()
                        }
                        _isCurrentlyDrawing.value = false
                        Log.d("HandTracking", "Both hands horizontal open - drawing disabled")
                    }
                } else {
                    // Not both hands detected, hide UI but keep all drawings
                    _fingerTipPosition.value = null

                    // If we were drawing, save the current path
                    if (_isCurrentlyDrawing.value && _currentPath.value.isNotEmpty()) {
                        _drawingPaths.value = _drawingPaths.value + listOf(_currentPath.value)
                        _currentPath.value = emptyList()
                    }
                    _isCurrentlyDrawing.value = false

                    Log.d(
                        "HandTracking",
                        "Missing hands - Left: $foundLeftHand, Right: $foundRightHand (drawings preserved)"
                    )
                }

            } else {
                _fingerTipPosition.value = null
            }
        } else {
            _fingerTipPosition.value = null
        }
    }

    fun checkIfHandIsOpen(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Boolean {
        // Much better hand open detection using finger joint relationships
        // MediaPipe hand landmarks:
        // Thumb: 1(knuckle), 2(middle), 3(joint), 4(tip)  
        // Index: 5(knuckle), 6(middle), 7(joint), 8(tip)
        // Middle: 9(knuckle), 10(middle), 11(joint), 12(tip)
        // Ring: 13(knuckle), 14(middle), 15(joint), 16(tip)
        // Pinky: 17(knuckle), 18(middle), 19(joint), 20(tip)

        var openFingers = 0

        // Check thumb (special case - compare x-coordinates for left/right instead of y)
        val thumbTip = landmarks[4]
        val thumbJoint = landmarks[3]
        val thumbMiddle = landmarks[2]

        // For thumb, check if tip is further from palm than the joint
        val thumbTipX = thumbTip.x() * imageWidth
        val thumbJointX = thumbJoint.x() * imageWidth
        val thumbMiddleX = thumbMiddle.x() * imageWidth

        // Thumb is extended if tip is further from center than joint
        val wrist = landmarks[0]
        val wristX = wrist.x() * imageWidth
        val isThumbOpen =
            kotlin.math.abs(thumbTipX - wristX) > kotlin.math.abs(thumbJointX - wristX)
        if (isThumbOpen) openFingers++

        Log.d(
            "HandTracking",
            "Thumb - tip: ${thumbTipX.toInt()}, joint: ${thumbJointX.toInt()}, wrist: ${wristX.toInt()}, open: $isThumbOpen"
        )

        // Check other fingers (index, middle, ring, pinky)
        val fingerData = listOf(
            Triple(8, 7, 6),   // Index: tip, joint, middle
            Triple(12, 11, 10), // Middle: tip, joint, middle  
            Triple(16, 15, 14), // Ring: tip, joint, middle
            Triple(20, 19, 18)  // Pinky: tip, joint, middle
        )

        for ((tipIndex, jointIndex, middleIndex) in fingerData) {
            val tip = landmarks[tipIndex]
            val joint = landmarks[jointIndex]
            val middle = landmarks[middleIndex]

            val tipY = tip.y() * imageHeight
            val jointY = joint.y() * imageHeight
            val middleY = middle.y() * imageHeight

            // Finger is extended if tip is above joint AND joint is above middle
            val isFingerExtended = tipY < jointY && jointY < middleY
            if (isFingerExtended) openFingers++

            Log.d(
                "HandTracking",
                "Finger $tipIndex - tip: ${tipY.toInt()}, joint: ${jointY.toInt()}, middle: ${middleY.toInt()}, extended: $isFingerExtended"
            )
        }

        // Hand is open if at least 3 fingers are extended
        val isOpen = openFingers >= 3
        Log.d("HandTracking", "Hand gesture - open fingers: $openFingers/5, isOpen: $isOpen")

        return isOpen
    }

    fun checkIfIndexFingerOnly(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Boolean {
        // Check if ONLY the index finger is extended while other fingers are closed

        // Check index finger - should be extended
        val indexTip = landmarks[8]
        val indexJoint = landmarks[7]
        val indexMiddle = landmarks[6]

        val indexTipY = indexTip.y() * imageHeight
        val indexJointY = indexJoint.y() * imageHeight
        val indexMiddleY = indexMiddle.y() * imageHeight

        val isIndexExtended = indexTipY < indexJointY && indexJointY < indexMiddleY

        if (!isIndexExtended) {
            Log.d("HandTracking", "Index finger not extended - not index-only gesture")
            return false
        }

        // Check that other fingers (middle, ring, pinky) are closed/curled
        val otherFingerData = listOf(
            Triple(12, 11, 10), // Middle: tip, joint, middle
            Triple(16, 15, 14), // Ring: tip, joint, middle
            Triple(20, 19, 18)  // Pinky: tip, joint, middle
        )

        var closedFingers = 0
        for ((tipIndex, jointIndex, middleIndex) in otherFingerData) {
            val tip = landmarks[tipIndex]
            val joint = landmarks[jointIndex]
            val middle = landmarks[middleIndex]

            val tipY = tip.y() * imageHeight
            val jointY = joint.y() * imageHeight
            val middleY = middle.y() * imageHeight

            // More strict check - finger is closed if tip is below joint by a meaningful amount
            // AND the joint is not significantly above the middle joint (finger curled)
            val isTipBelowJoint = tipY > (jointY + 5) // Tip must be at least 5 pixels below joint
            val isJointReasonableToMiddle =
                jointY >= (middleY - 15) // Joint not too far above middle
            val isFingerClosed = isTipBelowJoint && isJointReasonableToMiddle

            if (isFingerClosed) closedFingers++

            Log.d(
                "HandTracking",
                "Finger $tipIndex closed check - tip: ${tipY.toInt()}, joint: ${jointY.toInt()}, middle: ${middleY.toInt()}, " +
                        "tipBelowJoint: $isTipBelowJoint, jointReasonable: $isJointReasonableToMiddle, closed: $isFingerClosed"
            )
        }

        // Check thumb separately - thumb should not be extended outward
        val thumbTip = landmarks[4]
        val thumbJoint = landmarks[3]
        val wrist = landmarks[0]

        val thumbTipX = thumbTip.x() * imageWidth
        val thumbJointX = thumbJoint.x() * imageWidth
        val wristX = wrist.x() * imageWidth

        // Thumb is "closed" if it's not extended far from the palm
        val thumbDistance = kotlin.math.abs(thumbTipX - wristX)
        val thumbJointDistance = kotlin.math.abs(thumbJointX - wristX)
        val isThumbClosed = thumbDistance <= (thumbJointDistance + 20) // Small tolerance

        Log.d(
            "HandTracking",
            "Thumb check - tip dist: $thumbDistance, joint dist: $thumbJointDistance, closed: $isThumbClosed"
        )

        // All 3 other fingers should be closed AND thumb should not be extended
        val isIndexOnly = closedFingers >= 3 && isThumbClosed

        Log.d(
            "HandTracking",
            "Index finger only check - index extended: $isIndexExtended, other fingers closed: $closedFingers/3, thumb closed: $isThumbClosed, result: $isIndexOnly"
        )

        return isIndexOnly
    }

    fun checkIfHandIsHorizontalOpenPalm(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Boolean {
        // First check if hand is open (at least 3 fingers extended)
        if (!checkIfHandIsOpen(landmarks, imageWidth, imageHeight)) {
            return false
        }

        // Check palm orientation using cross product
        val wrist = landmarks[0]
        val indexBase = landmarks[5]
        val pinkyBase = landmarks[17]

        // Convert to pixel coordinates
        val wristX = wrist.x() * imageWidth
        val wristY = wrist.y() * imageHeight
        val wristZ = wrist.z() * imageWidth  // Using width as depth scale

        val indexBaseX = indexBase.x() * imageWidth
        val indexBaseY = indexBase.y() * imageHeight
        val indexBaseZ = indexBase.z() * imageWidth

        val pinkyBaseX = pinkyBase.x() * imageWidth
        val pinkyBaseY = pinkyBase.y() * imageHeight
        val pinkyBaseZ = pinkyBase.z() * imageWidth

        // Create vectors from wrist to index base and wrist to pinky base
        val v1x = indexBaseX - wristX
        val v1y = indexBaseY - wristY
        val v1z = indexBaseZ - wristZ

        val v2x = pinkyBaseX - wristX
        val v2y = pinkyBaseY - wristY
        val v2z = pinkyBaseZ - wristZ

        // Cross product to get palm normal vector
        val normalX = v1y * v2z - v1z * v2y
        val normalY = v1z * v2x - v1x * v2z
        val normalZ = v1x * v2y - v1y * v2x

        // Normalize the vector
        val magnitude = kotlin.math.sqrt(normalX * normalX + normalY * normalY + normalZ * normalZ)
        if (magnitude == 0.0f) return false

        val normalizedZ = normalZ / magnitude

        // More lenient check: lower threshold from -0.3 to -0.1 and add alternative method
        val isHorizontalByNormal = normalizedZ < -0.1 // More lenient palm facing camera/up

        // Also try the opposite direction in case of mirroring issues
        val isHorizontalByNormalFlipped =
            normalizedZ > 0.1 // Palm facing away but might be mirrored

        // Alternative method: Check if fingertips are roughly at the same Y level as wrist
        // This catches cases where the normal vector calculation might be off
        val fingertipY = landmarks[8].y() * imageHeight // Index finger tip
        val middleTipY = landmarks[12].y() * imageHeight // Middle finger tip
        val ringTipY = landmarks[16].y() * imageHeight // Ring finger tip
        val wristYCoord = wrist.y() * imageHeight

        // Check if fingertips are not significantly above or below wrist (horizontal-ish)
        val avgFingertipY = (fingertipY + middleTipY + ringTipY) / 3
        val verticalDistance = kotlin.math.abs(avgFingertipY - wristYCoord)
        val handLength =
            kotlin.math.sqrt((indexBaseX - wristX) * (indexBaseX - wristX) + (indexBaseY - wristY) * (indexBaseY - wristY))
        val isHorizontalByLevel =
            verticalDistance < handLength * 0.5f // Increased from 40% to 50% for more leniency

        val isHorizontal =
            isHorizontalByNormal || isHorizontalByNormalFlipped || isHorizontalByLevel

        Log.d(
            "HandTracking",
            "Palm orientation - normalZ: $normalizedZ, byNormal: $isHorizontalByNormal, " +
                    "byNormalFlipped: $isHorizontalByNormalFlipped, " +
                    "verticalDist: $verticalDistance, handLength: $handLength, byLevel: $isHorizontalByLevel, " +
                    "final: $isHorizontal"
        )

        return isHorizontal
    }

    fun getPalmDirection(
        landmarks: List<NormalizedLandmark>,
        imageWidth: Int,
        imageHeight: Int
    ): Pair<Float, Float> {
        // Calculate palm direction vector from wrist to middle finger base
        val wrist = landmarks[0]
        val middleBase = landmarks[9]

        val wristX = wrist.x() * imageWidth
        val wristY = wrist.y() * imageHeight
        val middleBaseX = middleBase.x() * imageWidth
        val middleBaseY = middleBase.y() * imageHeight

        // Direction vector (pointing from wrist toward fingers)
        val directionX = middleBaseX - wristX
        val directionY = middleBaseY - wristY

        // Normalize
        val magnitude = kotlin.math.sqrt(directionX * directionX + directionY * directionY)
        if (magnitude == 0.0f) return Pair(0f, 0f)

        return Pair(directionX / magnitude, directionY / magnitude)
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    FingerdrawingTheme {
        Greeting("Android")
    }
}

@Composable
fun CameraScreen(
    fingerTipPosition: State<Offset?>,
    drawingPaths: State<List<List<Offset>>>,
    isHandOpen: State<Boolean>,
    isLeftHand: State<Boolean>,
    isHorizontalOpenPalm: State<Boolean>,
    palmCenter: State<Offset?>,
    palmDirection: State<Pair<Float, Float>?>,
    leftHandHorizontalOpen: State<Boolean>,
    rightHandHorizontalOpen: State<Boolean>,
    leftPalmCenter: State<Offset?>,
    rightPalmCenter: State<Offset?>,
    leftPalmDirection: State<Pair<Float, Float>?>,
    rightPalmDirection: State<Pair<Float, Float>?>,
    previewSize: State<Pair<Int, Int>?>,
    isCurrentlyDrawing: State<Boolean>,
    currentPath: State<List<Offset>>
) {
    var hasCameraPermission by remember { mutableStateOf(false) }
    val context = LocalContext.current

    val cameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    LaunchedEffect(Unit) {
        val permissionGranted = ContextCompat.checkSelfPermission(
            context,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (permissionGranted) {
            hasCameraPermission = true
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    if (hasCameraPermission) {
        Box(modifier = Modifier.fillMaxSize()) {
            CameraPreview()
            // Overlay Canvas for fingertip position
            FingerTipOverlay(
                fingerTipPosition = fingerTipPosition.value,
                drawingPaths = drawingPaths.value,
                isHandOpen = isHandOpen.value,
                isLeftHand = isLeftHand.value,
                isHorizontalOpenPalm = isHorizontalOpenPalm.value,
                palmCenter = palmCenter.value,
                palmDirection = palmDirection.value,
                leftHandHorizontalOpen = leftHandHorizontalOpen.value,
                rightHandHorizontalOpen = rightHandHorizontalOpen.value,
                leftPalmCenter = leftPalmCenter.value,
                rightPalmCenter = rightPalmCenter.value,
                leftPalmDirection = leftPalmDirection.value,
                rightPalmDirection = rightPalmDirection.value,
                previewSize = previewSize.value,
                isCurrentlyDrawing = isCurrentlyDrawing.value,
                currentPath = currentPath.value
            )
        }
    } else {
        // Show loading or permission denied message
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("Camera permission required for hand tracking")
        }
    }
}

@Composable
fun CameraPreview() {
    val lifecycleOwner = LocalLifecycleOwner.current
    val context = LocalContext.current

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()
                val preview = androidx.camera.core.Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build()
                    .also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                // Initialize HandLandmarkerHelper on background thread
                val activity = ctx as MainActivity
                activity.backgroundExecutor.execute {
                    activity.handLandmarkerHelper = HandLandmarkerHelper(
                        context = ctx,
                        runningMode = RunningMode.LIVE_STREAM,
                        currentDelegate = HandLandmarkerHelper.DELEGATE_CPU,
                        maxNumHands = 2,  // Enable detection of 2 hands!
                        handLandmarkerHelperListener = activity
                    )
                    activity.isHandLandmarkerInitialized = true
                }

                val imageAnalyzer = ImageAnalysis.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                    .also {
                        it.setAnalyzer(activity.backgroundExecutor) { imageProxy ->
                            detectHand(imageProxy, activity)
                        }
                    }

                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalyzer
                    )
                } catch (exc: Exception) {
                    Log.e("CameraPreview", "Use case binding failed", exc)
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}

private fun detectHand(imageProxy: ImageProxy, activity: MainActivity) {
    if (activity.isHandLandmarkerInitialized) {
        activity.handLandmarkerHelper.detectLiveStream(
            imageProxy = imageProxy,
            isFrontCamera = true  // We ARE using front camera!
        )
    } else {
        imageProxy.close()
    }
}

// Draws a finger tip overlay on top of camera preview
@Composable
fun FingerTipOverlay(
    fingerTipPosition: Offset?,
    drawingPaths: List<List<Offset>>,
    isHandOpen: Boolean,
    isLeftHand: Boolean,
    isHorizontalOpenPalm: Boolean,
    palmCenter: Offset?,
    palmDirection: Pair<Float, Float>?,
    leftHandHorizontalOpen: Boolean,
    rightHandHorizontalOpen: Boolean,
    leftPalmCenter: Offset?,
    rightPalmCenter: Offset?,
    leftPalmDirection: Pair<Float, Float>?,
    rightPalmDirection: Pair<Float, Float>?,
    previewSize: Pair<Int, Int>?,
    isCurrentlyDrawing: Boolean,
    currentPath: List<Offset>,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier.fillMaxSize()) {
        // Canvas for drawing paths and finger indicators
        Canvas(modifier = Modifier.fillMaxSize()) {
            Log.d(
                "HandTracking",
                "FingerTipOverlay - isLeftHand: $isLeftHand, fingerTipPosition: $fingerTipPosition, isHandOpen: $isHandOpen"
            )

            // Always draw existing paths, regardless of hand detection
            if (previewSize != null && previewSize.first > 0 && previewSize.second > 0) {
                val scaleX = size.width / previewSize.first
                val scaleY = size.height / previewSize.second

                // Calculate the actual preview area (maintaining aspect ratio)
                val inputAspectRatio = previewSize.first.toFloat() / previewSize.second.toFloat()
                val displayAspectRatio = size.width / size.height
                val actualPreviewWidth = size.height * inputAspectRatio
                val xOffset = (size.width - actualPreviewWidth) / 2
                val actualScaleX = actualPreviewWidth / previewSize.first

                // Draw the drawing paths (always visible)
                for (path in drawingPaths) {
                    if (path.size > 1) {
                        val pathToDraw = Path()
                        var prev: Offset? = null
                        for ((index, point) in path.withIndex()) {
                            val px = xOffset + (point.x * actualScaleX)
                            val py = point.y * scaleY
                            val current = Offset(px, py)
                            if (index == 0) {
                                pathToDraw.moveTo(current.x, current.y)
                            } else {
                                prev?.let { prevPoint ->
                                    val mx = (prevPoint.x + current.x) / 2
                                    val my = (prevPoint.y + current.y) / 2
                                    pathToDraw.quadraticTo(prevPoint.x, prevPoint.y, mx, my)
                                }
                            }
                            prev = current
                        }
                        prev?.let { last ->
                            pathToDraw.lineTo(last.x, last.y)
                        }
                        drawPath(
                            path = pathToDraw,
                            color = Color.Red,
                            style = Stroke(width = 10f, cap = StrokeCap.Round)
                        )
                    }
                }

                // Draw the current drawing path (always visible)
                if (currentPath.size > 1) {
                    val path = Path()
                    var prev: Offset? = null
                    for ((index, point) in currentPath.withIndex()) {
                        val px = xOffset + (point.x * actualScaleX)
                        val py = point.y * scaleY
                        val current = Offset(px, py)
                        if (index == 0) {
                            path.moveTo(current.x, current.y)
                        } else {
                            prev?.let { prevPoint ->
                                val mx = (prevPoint.x + current.x) / 2
                                val my = (prevPoint.y + current.y) / 2
                                path.quadraticTo(prevPoint.x, prevPoint.y, mx, my)
                            }
                        }
                        prev = current
                    }
                    prev?.let { last ->
                        path.lineTo(last.x, last.y)
                    }
                    drawPath(
                        path = path,
                        color = Color.Red,
                        style = Stroke(width = 10f, cap = StrokeCap.Round)
                    )
                }

                // Check if both hands are in horizontal open palm mode
                val bothHandsHorizontalOpen = leftHandHorizontalOpen && rightHandHorizontalOpen

                // Only show palm center indicators and skip finger drawing if both hands are horizontal open
                if (bothHandsHorizontalOpen) {
                    // Draw left palm center indicator 
                    if (leftPalmCenter != null) {
                        val x = xOffset + (leftPalmCenter.x * actualScaleX)
                        val y = leftPalmCenter.y * scaleY

                        drawCircle(
                            color = Color.Blue,
                            radius = 8f,
                            center = Offset(x, y)
                        )
                        Log.d("HandTracking", "Drawing left palm center at ($x, $y)")
                    }

                    // Draw right palm center indicator
                    if (rightPalmCenter != null) {
                        val x = xOffset + (rightPalmCenter.x * actualScaleX)
                        val y = rightPalmCenter.y * scaleY

                        drawCircle(
                            color = Color.Blue,
                            radius = 8f,
                            center = Offset(x, y)
                        )
                        Log.d("HandTracking", "Drawing right palm center at ($x, $y)")
                    }

                    Log.d("HandTracking", "Both hands horizontal open - showing only fire effects")

                    // Early return - don't show finger tracking dots
                    return@Canvas
                }

                // Only draw the fingertip indicator if hands are detected and not both hands horizontal open
                if (!bothHandsHorizontalOpen && fingerTipPosition != null) {
                    val x = xOffset + (fingerTipPosition.x * actualScaleX)
                    val y = fingerTipPosition.y * scaleY

                    Log.d(
                        "HandTracking", "X-coordinate fix - " +
                                "InputAspect: $inputAspectRatio, DisplayAspect: $displayAspectRatio, " +
                                "ActualPreviewWidth: $actualPreviewWidth, XOffset: $xOffset, " +
                                "ActualScaleX: $actualScaleX, " +
                                "FingerTip adjusted: ($x, $y)"
                    )

                    drawCircle(
                        color = if (isHandOpen) Color.Green else Color.Red,
                        radius = 20f,
                        center = Offset(x, y)
                    )
                    Log.d("HandTracking", "Drawing dot for RIGHT hand - isHandOpen: $isHandOpen")
                }
            }
        }

        // Lottie fire animation for horizontal open palm
        if (previewSize != null) {
            val configuration = LocalConfiguration.current
            val screenWidthDp = configuration.screenWidthDp
            val screenHeightDp = configuration.screenHeightDp

            // Use the same scaling logic as the Canvas
            val inputAspectRatio = previewSize.first.toFloat() / previewSize.second.toFloat()
            val actualPreviewWidth = screenHeightDp * inputAspectRatio
            val xOffset = (screenWidthDp - actualPreviewWidth) / 2
            val actualScaleX = actualPreviewWidth / previewSize.first.toFloat()
            val scaleY = screenHeightDp.toFloat() / previewSize.second.toFloat()

            // Left hand fire animation
            if (leftHandHorizontalOpen && leftPalmCenter != null && leftPalmDirection != null && rightHandHorizontalOpen) {
                // Convert palm center to screen coordinates
                val palmScreenX = xOffset + (leftPalmCenter.x * actualScaleX)
                val palmScreenY = leftPalmCenter.y * scaleY

                // Use palm direction to position fire perpendicular to palm surface
                val palmDirection = leftPalmDirection
                val perpendicularX = palmDirection.second
                val perpendicularY = -palmDirection.first

                // Scale the perpendicular vector for fire distance (80 pixels away from palm)
                val fireDistance = 80f
                val fireOffsetX = perpendicularX * fireDistance
                val fireOffsetY = perpendicularY * fireDistance

                // Calculate rotation angle from palm direction vector (direction fingers are pointing)
                val angleRadians = kotlin.math.atan2(palmDirection.second, palmDirection.first)
                val angleDegrees = Math.toDegrees(angleRadians.toDouble()).toFloat()

                val composition by rememberLottieComposition(LottieCompositionSpec.Asset("flame-animation.json"))
                val progress by animateLottieCompositionAsState(
                    composition,
                    iterations = LottieConstants.IterateForever,
                    speed = 1f
                )

                LottieAnimation(
                    composition = composition,
                    progress = { progress },
                    modifier = Modifier
                        .size(100.dp)
                        .offset(
                            x = (palmScreenX + fireOffsetX - 50).dp, // Center the 100dp animation
                            y = (palmScreenY + fireOffsetY - 50).dp  // Center the 100dp animation
                        )
                        .rotate(angleDegrees)
                )

                Log.d(
                    "HandTracking",
                    "LEFT HAND Fire - Palm direction: (${palmDirection.first}, ${palmDirection.second}), " +
                            "Fire position: (${palmScreenX + fireOffsetX}, ${palmScreenY + fireOffsetY}), " +
                            "Angle degrees: $angleDegrees"
                )
            }

            // Right hand fire animation
            if (rightHandHorizontalOpen && rightPalmCenter != null && rightPalmDirection != null && leftHandHorizontalOpen) {
                // Convert palm center to screen coordinates
                val palmScreenX = xOffset + (rightPalmCenter.x * actualScaleX)
                val palmScreenY = rightPalmCenter.y * scaleY

                // Use palm direction to position fire perpendicular to palm surface
                val palmDirection = rightPalmDirection
                val perpendicularX = palmDirection.second   // Flip back to same as left hand
                val perpendicularY = -palmDirection.first   // Change back to same as left hand

                // Scale the perpendicular vector for fire distance (80 pixels away from palm)
                val fireDistance = 80f
                val fireOffsetX = perpendicularX * fireDistance
                val fireOffsetY = perpendicularY * fireDistance

                // Calculate rotation angle from palm direction vector (direction fingers are pointing)
                val angleRadians = kotlin.math.atan2(palmDirection.second, palmDirection.first)
                val angleDegrees = Math.toDegrees(angleRadians.toDouble()).toFloat()

                val composition by rememberLottieComposition(LottieCompositionSpec.Asset("flame-animation.json"))
                val progress by animateLottieCompositionAsState(
                    composition,
                    iterations = LottieConstants.IterateForever,
                    speed = 1f
                )

                LottieAnimation(
                    composition = composition,
                    progress = { progress },
                    modifier = Modifier
                        .size(100.dp)
                        .offset(
                            x = (palmScreenX + fireOffsetX - 50).dp, // Center the 100dp animation
                            y = (palmScreenY + fireOffsetY - 50).dp  // Center the 100dp animation
                        )
                        .rotate(angleDegrees)
                )

                Log.d(
                    "HandTracking",
                    "RIGHT HAND Fire - Palm direction: (${palmDirection.first}, ${palmDirection.second}), " +
                            "Fire position: (${palmScreenX + fireOffsetX}, ${palmScreenY + fireOffsetY}), " +
                            "Angle degrees: $angleDegrees"
                )
            }
        }
    }
}
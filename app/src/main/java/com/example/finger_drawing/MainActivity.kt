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
import androidx.compose.ui.graphics.ColorFilter
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
import com.airbnb.lottie.LottieProperty
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
                            val isOpen = checkIfHandIsOpen(
                                landmarks,
                                resultBundle.inputImageWidth,
                                resultBundle.inputImageHeight
                            )

                            // Check if hand is horizontal open palm
                            val isHorizontalOpen = checkIfHandIsHorizontalOpenPalm(
                                landmarks,
                                resultBundle.inputImageWidth,
                                resultBundle.inputImageHeight
                            )

                            Log.d(
                                "HandTracking",
                                "Hand $handIndex - Finger tip: (${fingerTipX.toInt()}, ${fingerTipY.toInt()}), " +
                                        "Handedness: $handenessLabel, Left: $isLeft, Open: $isOpen, Horizontal Open: $isHorizontalOpen"
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
                                _isHorizontalOpenPalm.value = true
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
                                _palmCenter.value = Offset(centerX, centerY)

                                val palmDirection = getPalmDirection(
                                    landmarks,
                                    resultBundle.inputImageWidth,
                                    resultBundle.inputImageHeight
                                )
                                _palmDirection.value = palmDirection
                            } else {
                                _isHorizontalOpenPalm.value = false
                                _palmCenter.value = null
                                _palmDirection.value = null
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
                            _drawingPaths.value = _drawingPaths.value + listOf(_currentPath.value)
                            _currentPath.value = emptyList()
                        }
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

        // Check if palm is facing up (normal z-component should be negative for front camera)
        // Also check that the normal is sufficiently vertical (not too tilted)
        val isHorizontal = normalizedZ < -0.3 // Palm facing camera/up

        Log.d(
            "HandTracking",
            "Palm orientation - normalZ: $normalizedZ, isHorizontal: $isHorizontal"
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

                // Draw palm center indicator for horizontal open palm (but not the fire - that's handled by Lottie)
                if (isHorizontalOpenPalm && palmCenter != null) {
                    val x = xOffset + (palmCenter.x * actualScaleX)
                    val y = palmCenter.y * scaleY

                    // Draw palm center indicator
                    drawCircle(
                        color = Color.Blue,
                        radius = 8f,
                        center = Offset(x, y)
                    )

                    Log.d("HandTracking", "Drawing palm center at ($x, $y)")
                }

                // Only draw the fingertip indicator if hands are detected and no horizontal palm
                if (!isHorizontalOpenPalm && !isLeftHand && fingerTipPosition != null) {
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
        if (isHorizontalOpenPalm && palmCenter != null && previewSize != null) {
            val configuration = LocalConfiguration.current
            val screenWidthDp = configuration.screenWidthDp
            val screenHeightDp = configuration.screenHeightDp

            // Use the same scaling logic as the Canvas
            val inputAspectRatio = previewSize.first.toFloat() / previewSize.second.toFloat()
            val actualPreviewWidth = screenHeightDp * inputAspectRatio
            val xOffset = (screenWidthDp - actualPreviewWidth) / 2
            val actualScaleX = actualPreviewWidth / previewSize.first
            val scaleY = screenHeightDp.toFloat() / previewSize.second

            // Convert palm center to screen coordinates
            val palmScreenX = xOffset + (palmCenter.x * actualScaleX)
            val palmScreenY = palmCenter.y * scaleY

            // Since phone is horizontal and palm appears sideways:
            // Use palm direction to position fire perpendicular to palm surface
            if (palmDirection != null) {
                // Palm direction vector points from wrist toward fingers
                // Perpendicular vector (rotated 90 degrees) points away from palm surface
                // We want it to point "above" the palm, so we flip the direction
                val perpendicularX = palmDirection.second  // Flipped: use +dy instead of -dy
                val perpendicularY = -palmDirection.first  // Flipped: use -dx instead of +dx

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
                        .rotate(angleDegrees),
                    dynamicProperties = rememberLottieDynamicProperties(
                        rememberLottieDynamicProperty(
                            property = LottieProperty.COLOR,
                            value = Color(0xFFFFA500), // Orange color
                            keyPath = arrayOf("**")
                        )
                    )
                )

                Log.d(
                    "HandTracking",
                    "Palm direction: (${palmDirection.first}, ${palmDirection.second}), " +
                            "Perpendicular: ($perpendicularX, $perpendicularY), " +
                            "Fire offset: ($fireOffsetX, $fireOffsetY), " +
                            "Fire position: (${palmScreenX + fireOffsetX}, ${palmScreenY + fireOffsetY}), " +
                            "Angle degrees: $angleDegrees"
                )
            }
        }
    }
}
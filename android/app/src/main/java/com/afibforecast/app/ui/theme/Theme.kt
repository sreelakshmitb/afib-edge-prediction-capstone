package com.afibforecast.app.ui.theme

import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val Cream = Color(0xFFF8F3EA)
val Paper = Color(0xFFFFFBF4)
val Olive = Color(0xFFA7B86D)
val OliveDeep = Color(0xFF66773A)
val Orange = Color(0xFFF08022)
val AmberSoft = Color(0xFFFFD07A)
val Brown = Color(0xFF624638)
val BrownMuted = Color(0xFF846C5E)
val ActionBrown = Color(0xFF8B6655)
val Lavender = Color(0xFFB8A7E8)
val Line = Color(0xFFE7DDD1)
val SagePale = Color(0xFFE4E9CF)
val ErrorRed = Color(0xFFA34234)
val RiskPale = Color(0xFFFBE8E4)

private val AppColors = lightColorScheme(
    primary = OliveDeep,
    onPrimary = Paper,
    secondary = Orange,
    onSecondary = Brown,
    background = Cream,
    onBackground = Brown,
    surface = Paper,
    onSurface = Brown,
    surfaceVariant = SagePale,
    onSurfaceVariant = BrownMuted,
    outline = Line,
    error = ErrorRed,
)

private val AppTypography = Typography(
    displayLarge = TextStyle(fontFamily = FontFamily.Serif, fontWeight = FontWeight.Bold, fontSize = 46.sp, lineHeight = 48.sp),
    displayMedium = TextStyle(fontFamily = FontFamily.Serif, fontWeight = FontWeight.Bold, fontSize = 38.sp, lineHeight = 41.sp),
    headlineLarge = TextStyle(fontFamily = FontFamily.Serif, fontWeight = FontWeight.Bold, fontSize = 32.sp, lineHeight = 36.sp),
    headlineMedium = TextStyle(fontFamily = FontFamily.Serif, fontWeight = FontWeight.Bold, fontSize = 27.sp, lineHeight = 31.sp),
    titleLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 22.sp, lineHeight = 28.sp),
    titleMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 18.sp, lineHeight = 24.sp),
    bodyLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 18.sp, lineHeight = 26.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Medium, fontSize = 16.sp, lineHeight = 23.sp),
    labelLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 16.sp, lineHeight = 20.sp),
    labelMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 14.sp, lineHeight = 18.sp),
)

@Composable
fun AFibForecastTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = AppColors, typography = AppTypography, content = content)
}

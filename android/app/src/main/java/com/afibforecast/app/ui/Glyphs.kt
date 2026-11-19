package com.afibforecast.app.ui

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke

enum class Glyph { WAVE, HISTORY, INFO, CONNECTION, PLAY, PAUSE, HEART, CHECK, BACK, REFRESH, EMERGENCY_CONTACT }

@Composable
fun AppGlyph(glyph: Glyph, color: Color, modifier: Modifier = Modifier) {
    Canvas(modifier) {
        val w = size.width
        val h = size.height
        val stroke = Stroke(width = w * .085f, cap = StrokeCap.Round, join = StrokeJoin.Round)
        when (glyph) {
            Glyph.WAVE -> {
                val p = Path().apply {
                    moveTo(w * .06f, h * .55f); lineTo(w * .28f, h * .55f)
                    lineTo(w * .39f, h * .26f); lineTo(w * .53f, h * .78f)
                    lineTo(w * .66f, h * .42f); lineTo(w * .76f, h * .55f); lineTo(w * .95f, h * .55f)
                }
                drawPath(p, color, style = stroke)
            }
            Glyph.HISTORY -> {
                drawArc(color, 42f, 280f, false, topLeft = Offset(w*.12f,h*.12f), size = Size(w*.76f,h*.76f), style = stroke)
                drawLine(color, Offset(w*.5f,h*.5f), Offset(w*.5f,h*.28f), stroke.width, StrokeCap.Round)
                drawLine(color, Offset(w*.5f,h*.5f), Offset(w*.68f,h*.62f), stroke.width, StrokeCap.Round)
                val p=Path().apply{moveTo(w*.12f,h*.16f);lineTo(w*.14f,h*.39f);lineTo(w*.34f,h*.27f)}
                drawPath(p,color,style=stroke)
            }
            Glyph.INFO -> {
                drawCircle(color, w*.38f, Offset(w*.5f,h*.5f), style=stroke)
                drawCircle(color, w*.045f, Offset(w*.5f,h*.29f))
                drawLine(color,Offset(w*.5f,h*.44f),Offset(w*.5f,h*.7f),stroke.width,StrokeCap.Round)
            }
            Glyph.EMERGENCY_CONTACT -> {
                drawCircle(color, w * .125f, Offset(w * .44f, h * .30f), style = stroke)
                val shoulders = Path().apply {
                    moveTo(w * .16f, h * .75f)
                    cubicTo(w * .16f, h * .47f, w * .72f, h * .47f, w * .72f, h * .75f)
                }
                drawPath(shoulders, color, style = stroke)
                drawCircle(color, w * .18f, Offset(w * .76f, h * .72f), style = stroke)
                drawLine(color, Offset(w * .76f, h * .63f), Offset(w * .76f, h * .81f), stroke.width, StrokeCap.Round)
                drawLine(color, Offset(w * .67f, h * .72f), Offset(w * .85f, h * .72f), stroke.width, StrokeCap.Round)
            }
            Glyph.CONNECTION -> {
                val d = minOf(w, h)
                val left = (w - d) / 2f
                val top = (h - d) / 2f
                val wifiStroke = Stroke(width = d * .085f, cap = StrokeCap.Round)
                fun wifiArc(x1: Float, y1: Float, cx1: Float, cy1: Float, cx2: Float, cy2: Float, x2: Float, y2: Float) {
                    val path = Path().apply {
                        moveTo(left + d * x1, top + d * y1)
                        cubicTo(left + d * cx1, top + d * cy1, left + d * cx2, top + d * cy2, left + d * x2, top + d * y2)
                    }
                    drawPath(path, color, style = wifiStroke)
                }
                wifiArc(.12f, .38f, .33f, .18f, .67f, .18f, .88f, .38f)
                wifiArc(.28f, .55f, .41f, .43f, .59f, .43f, .72f, .55f)
                wifiArc(.41f, .70f, .47f, .65f, .53f, .65f, .59f, .70f)
                drawCircle(color, d * .06f, Offset(left + d * .5f, top + d * .82f))
            }
            Glyph.PLAY -> {
                val p=Path().apply{moveTo(w*.35f,h*.22f);lineTo(w*.78f,h*.5f);lineTo(w*.35f,h*.78f);close()}
                drawPath(p,color)
            }
            Glyph.PAUSE -> {
                drawRoundRect(color,Offset(w*.27f,h*.2f),Size(w*.16f,h*.6f),cornerRadius=androidx.compose.ui.geometry.CornerRadius(w*.06f))
                drawRoundRect(color,Offset(w*.57f,h*.2f),Size(w*.16f,h*.6f),cornerRadius=androidx.compose.ui.geometry.CornerRadius(w*.06f))
            }
            Glyph.HEART -> {
                val p=Path().apply{moveTo(w*.5f,h*.82f);cubicTo(w*.35f,h*.69f,w*.12f,h*.54f,w*.12f,h*.31f);cubicTo(w*.12f,h*.11f,w*.39f,h*.08f,w*.5f,h*.27f);cubicTo(w*.61f,h*.08f,w*.88f,h*.11f,w*.88f,h*.31f);cubicTo(w*.88f,h*.54f,w*.65f,h*.69f,w*.5f,h*.82f);close()}
                drawPath(p,color,style=stroke)
            }
            Glyph.CHECK -> { drawLine(color,Offset(w*.18f,h*.52f),Offset(w*.42f,h*.74f),stroke.width,StrokeCap.Round);drawLine(color,Offset(w*.42f,h*.74f),Offset(w*.84f,h*.25f),stroke.width,StrokeCap.Round) }
            Glyph.BACK -> { drawLine(color,Offset(w*.68f,h*.18f),Offset(w*.32f,h*.5f),stroke.width,StrokeCap.Round);drawLine(color,Offset(w*.32f,h*.5f),Offset(w*.68f,h*.82f),stroke.width,StrokeCap.Round) }
            Glyph.REFRESH -> {
                drawArc(color,40f,280f,false,topLeft=Offset(w*.15f,h*.15f),size=Size(w*.70f,h*.70f),style=stroke)
                val arrow = Path().apply {
                    moveTo(w*.61f,h*.18f)
                    lineTo(w*.77f,h*.275f)
                    lineTo(w*.80f,h*.09f)
                }
                drawPath(arrow,color,style=stroke)
            }
        }
    }
}

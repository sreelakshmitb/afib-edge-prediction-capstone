package com.afibforecast.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build

class AFibForecastApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "verified_forecasts",
                "Verified live forecasts",
                NotificationManager.IMPORTANCE_DEFAULT,
            ).apply {
                description = "Research-prototype notifications for verified live model results"
            }
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }
}

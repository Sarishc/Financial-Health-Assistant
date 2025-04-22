import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:financial_health_assistant/main.dart';

void main() {
  testWidgets('App renders correctly', (WidgetTester tester) async {
    // Build our app and trigger a frame
    await tester.pumpWidget(const FinancialHealthAssistant());

    // Verify that our app is showing the correct content
    expect(find.text('Financial Health Assistant'), findsOneWidget);
    expect(find.text('Dashboard Screen'), findsOneWidget);
    
    // Test navigation to other screens
    await tester.tap(find.text('Transactions'));
    await tester.pumpAndSettle();
    expect(find.text('Transactions Screen'), findsOneWidget);
    
    await tester.tap(find.text('Forecasts'));
    await tester.pumpAndSettle();
    expect(find.text('Forecasts Screen'), findsOneWidget);
    
    await tester.tap(find.text('Recommendations'));
    await tester.pumpAndSettle();
    expect(find.text('Recommendations Screen'), findsOneWidget);
    
    // Go back to Dashboard
    await tester.tap(find.text('Dashboard'));
    await tester.pumpAndSettle();
    expect(find.text('Dashboard Screen'), findsOneWidget);
  });
}
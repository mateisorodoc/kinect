   M S B A M L   `   `   `  ����  DepthBasics-WPF5    /Microsoft.Samples.Kinect.DepthBasics.MainWindow   5   	   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   -   5          � Title$  Depth Basics��5   	   $	��735��6   $	��770��6+    � 	Resources 5         MediumGreyBrush �� Color  KinectPurpleBrush  KinectBlueBrush �� 
TargetType �� Property �� Value  SnapshotButton �� 
TargetType �� Orientation ��Name	 �� TextAlignment
 ��Name �� Triggers P� Property P� Value �� 
TargetName  SquareCheckBox �Name��%w
  &	        &	 #     &	 F     '�� i     &	 �     7  7  &	 �    7  7  �� 5   
   $ 	#ff6e6e6e��62   �� 5   
   $ 	#ff52318f��64   �� 5	   
   $ 	#ff00BCF2��62   �� 5
   
   " ��6   .�5      ��  �@J�SnapsToDevicePixels . 64   5      �� 5   
   " ��6   .�5      ��  � �� 5      �� 5      " ��6&   .R�5      � .+�5      �� $ 
Horizontal=�6)   
������ .+�5   "   �� $ SnapText��6,   #��z��6>   $
	 Left=�6_   $��Center=�6u   8��  $��15(�6�   � 5   "   $��9,0,0,0q�6'   .+�5   &   �� $
 
SnapNormal��6,   $��Images\SnapNormal.png��6@   $
��None=�6_   $��Center=�6n   �� 5   &   $
 	SnapHover��6,   $��Images\SnapHover.png��6?   $
��None=�6]   $��Center=�6l   $y�	Collapsed=�6�   5   #   5      5       5      P� 5       �@J�IsMouseOver . . �5   "   �� $ 
SnapNormal��6Q    � y�$ 	Collapsed=�6?   6Q   �� 5   "   $ 	SnapHover��6O    � y�$ Visible=�6?   6O   �� 5   "   8  $ SnapText��6h    � ��5       5"      5$      5%      �� 5&   
   " ��6   .�5'      ��  � �� 5(      �� 5)      " ��6&   .R�5*      � .+�5+      �� $ 
Horizontal=�6)   
������ .+�5,   "   � $ SquareCheckBoxChecked��6'   .+�5-   &   �� $
 CheckedNormal��6,   $��Images\CheckedNormal.png��6C   $
��None=�6e   $��Center=�6t   �� 5.   &   $
 CheckedHover��6,   $��Images\CheckedHover.png��6B   $
��None=�6c   $��Center=�6r   $y�	Collapsed=�6�   5/   #   � 50   "   $ SquareCheckBoxUnchecked��6'   $y�	Collapsed=�6H   .+�51   &   �� $
 UncheckedNormal��6,   $ ��Images\UncheckedNormal.png��6E   $
��None=�6i   $��Center=�6x   �� 52   &   $
 UncheckedHover��6,   $��Images\UncheckedHover.png��6D   $
��None=�6g   $��Center=�6v   $y�	Collapsed=�6�   53   #   �� 54   "   $ SquareCheckBoxText��6,   #��z��6H   $
	 Left=�6i   $��Center=�6   8��  $��15(�6�   $��9,0,0,0q�6�   55      56       57      P� 58       �@d�	IsChecked$ falseY�6;   . �59   "   �� $ SquareCheckBoxChecked��6Q    � y�$ 	Collapsed=�6?   6Q   �� 5:   "   $ SquareCheckBoxUnchecked��6O    � y�$ Visible=�6?   6O   5;      P� 5<       �@J�IsMouseOver . . �5=   "   �� $ CheckedNormal��6Q    � y�$ 	Collapsed=�6?   6Q   �� 5>   "   $ UncheckedNormal��6Q    � y�$ 	Collapsed=�6?   6Q   �� 5?   "   $ CheckedHover��6O    � y�$ Visible=�6?   6O   �� 5@   "   $ UncheckedHover��6O    � y�$ Visible=�6?   6O   �� 5A   "   8  $ SquareCheckBoxText��6h    � ��5B      5D      5F      5G      5H      .��5J      � -   5        $��
layoutGrid��5J      $��	10 0 10 0q�6    � RowDefinitions 5K   
   �� 5L      $
��Auto �6   �� 5M      �� 5N      $
��Auto �6   $��61.04��6*   �� 5O      $
��Auto �6   $��26��6*   .+�5Q   
   ]� $��0��6   $��0,0,0,20q�6!   .+�5R      �� $
��Left=�6   $��Images\Logo.png��6*   $
��Fill=�6C   $��32��6R   $��81��6^   $��0 10 0 5q�6i   �� 5S      $��Right=�6   $��0 0 -1 0q�6/   $��Bottom=�6A   #��[  6\   $��Segoe UI)�6�   $��18(�6�   Depth Basics6�   �� 5T      $��1��6   $��Images\Status.png��6$   $
��None=�6?   $��Center=�6N   $��0 0 0 5q�6k   5U      (� 5V   
   $��1��6    (� Stretch$ Uniform=�6   $��Center=�61   $	��750��6N   .��5W      �� -   5        $��Image��5W      $	��640��6!   $	��480��6-   5X      (� 5Y   
   $��1��6   $ Uniform=�6   $��Center=�61   $	��750��6N   .��5Z   
   �� -   5        $��RgbImage��5Z      $	��640��6    $	��480��6,   5[      (� 5\   
   $��1��6   $ Uniform=�6   $��Center=�61   $	��750��6N   .��5]      �� -   5        $��HeatmapImage��5]      $	��640��6(   $	��480��64   5^      �� 5`   
   -   5        $��	statusBar��5`   !   $��3��6   $��Center=�62   
�������#��[  6`   $��26��6�   .@�5a      �� $��0 0 0 10q�6   .��5b      �� -   5        $��statusBarText��5b      $��-1 0 0 0q�61   Kinect connected6U   5c      5d      �� 5e   
   -   5        $��
TiltSlider��5e       � Minimum$	 -27Q�5f   	    � Maximum$ 27Q�5g   	    � Value$ 0Q�5h   	    �� TickFrequency$ 1Q�5i   	    �� IsSnapToTickEnabled . $	��120��5k   	   $��Center=�5l   	   $��2��5m   	   $��Center=�5n   	   5p   	   �� 5r   
   $��-27°��6   $
��Left=�6   $��	267,0,0,0q�6:   $��2��6M   $��Center=�6Z   �� 5s   
   $��+27°��6   $
��Left=�6   $��	455,0,0,0q�6:   $��2��6M   $��Center=�6Z   �� 5t   
   -	   5        $��TiltValueLabel��5t      $	��0°��5u      $��Center=�5v      $��2��5w      $��Center=�5x      $��0,0,0,40q�5y      5{      5|      
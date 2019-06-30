//////////////////////////////////////////////////////////////////////////////////////
//
//   JavaGraphics source code (c) 2003, Wolfram Research, Inc. All rights reserved.
//
//   Author: Dale R. Horton
//
//////////////////////////////////////////////////////////////////////////////////////

// This is the class used by JavaGraphics to play sounds in the kernel.

package com.wolfram.viewer;

import javax.swing.*;
import java.applet.*;
import java.awt.*;
import java.awt.event.*;
import java.net.*;
import com.wolfram.jlink.*;

/* Comments:
Cannot toggle play button because there's no event listener 
to tell us when the sound has finished on it's own. 
*/

public class PlaySound extends MathJFrame implements ActionListener {

    AudioClip clip;
    JButton playButton, stopButton;

    public PlaySound(String filename, String icondir) {
        
        JPanel controlPanel = new JPanel();
        int w, h;
        int buffer=15, offset=0;
        
        try{
          /* get file */
          URL url = new URL("file:"+filename);
          clip = Applet.newAudioClip(url);
          
          /* create buttons */
          ImageIcon playpic = new ImageIcon(icondir+"/play.gif", "Play");
          playButton = new JButton(playpic);
          playButton.setBackground(Color.white);
          playButton.addActionListener(this);

          ImageIcon stoppic = new ImageIcon(icondir+"/stop.gif", "Stop");
          stopButton = new JButton(stoppic);
          stopButton.setBackground(Color.white);
          stopButton.addActionListener(this);
                
          controlPanel.add(playButton);
          controlPanel.add(stopButton);
          w = 150;
          h = 50;

        } catch (MalformedURLException e) {
        
          Label l = new Label("Invalid filename: "+filename);
          controlPanel.add(l);
          w = l.getWidth();
          h = l.getHeight();
        
        }
        
        /* put it all together */
        controlPanel.setBackground(Color.white); 
        getContentPane().add(controlPanel);
        setSize(w+2*buffer, h+2*buffer+offset);
        clip.play();
    }

    public void actionPerformed(ActionEvent event) {
        /* play button */
        Object source = event.getSource();
        if (source == playButton) {
            clip.play();
            stopButton.setEnabled(true); 
            return;
        }

        /* stop button */
        if (source == stopButton) {
            clip.stop();
            stopButton.setEnabled(false); 
            return;
        }
    }
}
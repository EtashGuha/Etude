//////////////////////////////////////////////////////////////////////////////////////
//
//   JavaGraphics source code (c) 2003, Wolfram Research, Inc. All rights reserved.
//
//   Author: Dale R. Horton
//
//////////////////////////////////////////////////////////////////////////////////////

// This is the class used by JavaGraphics to view animations in the kernel.

package com.wolfram.viewer;

import java.applet.Applet;
import java.awt.*;
import java.awt.event.*;
import java.net.*;
import java.awt.image.*;
import java.io.*;
import com.wolfram.jlink.*;
import javax.swing.*;

public class AnimateGIF extends MathJFrame implements ActionListener {

        Image img;
        Toolkit tk = Toolkit.getDefaultToolkit();
        ImageObserverSpy spy = new ImageObserverSpy(this);
        int buffer=15, offset=70;
        boolean playstatus=true;
        JButton playButton, stopButton;
        
        public AnimateGIF(String filename, String icondir) {
             int w, h;
             JPanel controlPanel = new JPanel();
        
             try{
                /* get file */
                img = tk.createImage(filename);
                
                MediaTracker mt = new MediaTracker(this);
                mt.addImage(img, 0);
                try{mt.waitForID(0);} catch (InterruptedException e) {}
                
                /* get image size */
                w = img.getWidth(this);
                if (w==-1) w=288;
                h = img.getHeight(this);
                if (h==-1) h=177;
          
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
                
                playButton.setEnabled(false); 
                stopButton.setEnabled(true); 

              } catch (Exception e) {
              
                Label l = new Label("Cannot display file: "+filename);
                controlPanel.add(l);
                w = l.getWidth();
                h = l.getHeight();
                
              }
              
              /* put it all together */
              controlPanel.setBackground(Color.white); 
              getContentPane().add(controlPanel);        
              setSize(w+2*buffer, h+2*buffer+offset);
              setBackground(Color.white);
              
        }

        public void paint(Graphics g)
        {       
                if (playstatus) {
                  /* buttons are not drawn correctly without calling super */
                  super.paint(g);
                  g.drawImage(img, buffer, buffer+offset, spy);
                }
        }
        public void update(Graphics g)
        {
                paint(g);
        }
        
        public void actionPerformed(ActionEvent event) {
        /* play button */
        Object source = event.getSource();
        if (source == playButton) {
            playstatus=true;
            playButton.setEnabled(false); 
            stopButton.setEnabled(true); 
            return;
        }

        /* stop button */
        if (source == stopButton) {
            playstatus=false;
            stopButton.setEnabled(false); 
            playButton.setEnabled(true); 
            return;
        }
    }
} 

/* Observe the image so we can insert pauses between the frames. */      
class ImageObserverSpy implements ImageObserver{

        ImageObserver obs;

        ImageObserverSpy(ImageObserver obs) {
          this.obs = obs;
        }
              
        public boolean imageUpdate(Image img, int info, int x, int y, int w, int h){
          if ((info & FRAMEBITS)!=0)
            try{
              Thread.sleep(250);
            } catch(InterruptedException e){}
          return obs.imageUpdate(img, info, x, y, w, h);
        }
     }

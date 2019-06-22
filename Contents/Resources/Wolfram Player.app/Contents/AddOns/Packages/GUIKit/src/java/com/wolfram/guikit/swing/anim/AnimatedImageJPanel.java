/*
 * @(#)AnimatedImageJPanel.java
 *
 * Copyright (c) 2005 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing.anim;

import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.LayoutManager;
import java.util.TimerTask;
import java.util.Vector;

import javax.swing.JPanel;

import com.wolfram.guikit.GUIKitUtils;


/**
 * AnimatedImageJPanel 
 */
public class AnimatedImageJPanel extends JPanel {
		
  private static final long serialVersionUID = -1187987971456711948L;
    
  private Dimension offDimension;
  private Graphics offGraphics;
  private Image offImage;
  
	private Vector images = new Vector();
  private int currentFrame = 0;
  
  private java.util.Timer animationTimer = null;
  private AnimationTimerTask animationTimerTask = null;
  
  private double framesPerSecond = 10;
  
	public AnimatedImageJPanel(LayoutManager layout, boolean isDoubleBuffered) {
		super(layout, isDoubleBuffered);
		}

	public AnimatedImageJPanel(LayoutManager layout) {
		this(layout, true);
		}

	public AnimatedImageJPanel(boolean isDoubleBuffered) {
		this(new FlowLayout(), isDoubleBuffered);
		}

	public AnimatedImageJPanel() {
		this(true);
		}
	
  public double getFramesPerSecond() {return framesPerSecond;}
  public void setFramesPerSecond(double framesPerSecond) {
    this.framesPerSecond = framesPerSecond;
    if (animationTimer != null) {
      resetTimer();
      }
    }
  
  public void resetTimer() {
    stop();
    start();
    }
  
	public void start() {
    if (animationTimer == null) {
      animationTimerTask = new AnimationTimerTask();
      animationTimer = new java.util.Timer();
      animationTimer.schedule(animationTimerTask, 10, Math.round(1000.0/framesPerSecond));
      }
    }
  
  public void stop() {
    if (animationTimer != null) {
      animationTimer.cancel();
      animationTimer = null;
      if (animationTimerTask != null) {
        animationTimerTask.cancel();
        animationTimerTask = null;
        }
      }
    }
  
  public void back() {
    step(-1);
    }
  
  public void forward() {
    step(1);
    }
  
  public void step() {
    step(1);
    }
  
  public void step(int delta) {
    currentFrame =  ((currentFrame + delta) % images.size());
    if (currentFrame < 0) currentFrame += images.size();
    repaint();
    }
  
  public int getFrameCount() {return images.size();}
  
  public int getFrame() {return currentFrame;}
  public void setFrame(int index) {
    currentFrame = index % images.size();
    if (currentFrame < 0) currentFrame += images.size();
    repaint();
    }
  
  public void addFrame(Image im) {
    if (im == null) return;
    images.add(im);
    if (animationTimer == null) setFrame(images.size()-1);
    }
  
  public void addFrameData(Object data) {
    addFrame( GUIKitUtils.createImage(data));
    }
	
  /**
   * Update a frame of animation.
   */
  public void update(Graphics g) {
    Dimension d = getSize();
  
    // Create the offscreen graphics context
    if ((offGraphics == null)
     || (d.width != offDimension.width)
     || (d.height != offDimension.height)) {
        offDimension = d;
        offImage = createImage(d.width, d.height);
        offGraphics = offImage.getGraphics();
       }
  
    // Paint the frame into the image
    paintFrame(offGraphics);
  
    // Paint the image onto the screen
    g.drawImage(offImage, 0, 0, null);
    }
    
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
    update(g);
		}
			
	public void paintFrame(Graphics g) {
    if (currentFrame < 0 || currentFrame >= images.size()) return;
    Image im = (Image)images.get(currentFrame);
    if (im == null) return;
    
    int left = (getWidth() - im.getWidth(this))/2;
    int top = (getHeight() - im.getHeight(this))/2;
		g.drawImage(im, left, top, this);
    }
    
  private class AnimationTimerTask extends TimerTask {
    public AnimationTimerTask() {
      }
    public void run() {
      step();
      }
    }
  
  }

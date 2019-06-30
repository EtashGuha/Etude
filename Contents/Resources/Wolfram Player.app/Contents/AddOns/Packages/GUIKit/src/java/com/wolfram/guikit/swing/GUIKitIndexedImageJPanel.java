/*
 * @(#)GUIKitIndexedImageJPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.image.*;

import javax.swing.JPanel;

import com.wolfram.jlink.Utils;

/**
 * GUIKitIndexedImageJPanel provides a simple
 * convenience JComponent for displaying indexed image data
 */
public class GUIKitIndexedImageJPanel extends JPanel {
		
    private static final long serialVersionUID = -1287447975459789948L;
    
	private BufferedImage image;
	
	private int imageWidth = -1;
	private int imageHeight = -1;
	private int imageTransparentPixel = -1;
	private boolean imageHasAlpha = false;
	private int[] imageColorComponents;
	private int imageColorMapSize = 0;
	private int imageGrayLevelMapSize = -1;
  
	private boolean needsImageCreate = false;
	
	private boolean imageGrid = false;
  private Color imageGridColor = Color.BLACK;
	private boolean centerImage = true;
	private boolean scaleImage = false;
	private boolean preserveImageAspectRatio = false;
	
	private int imageX = 0;
	private int imageY = 0;
	
	private int imagePixelSize = 1;
	
	public static final int DROP = 0;
	public static final int WRAP = 1;
	
	private int imagePixelOutOfBoundsMode = DROP;
	 
	private static RenderingHints renderHints;
	private static boolean requiresHints = false;
	
	static {
		// We do this so that MacOS X does not antialias the image in the panel
		renderHints = new RenderingHints(
		  RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
		requiresHints = Utils.isMacOSX();
		}
	
	public GUIKitIndexedImageJPanel(LayoutManager layout, boolean isDoubleBuffered) {
		super(layout, isDoubleBuffered);
		}

	public GUIKitIndexedImageJPanel(LayoutManager layout) {
		this(layout, true);
		}

	public GUIKitIndexedImageJPanel(boolean isDoubleBuffered) {
		this(new FlowLayout(), isDoubleBuffered);
		}

	public GUIKitIndexedImageJPanel() {
		this(true);
		}
	
	public Image getImage() {return image;}
	
	public int getImagePixelOutOfBoundsMode() {return imagePixelOutOfBoundsMode;}
	public void setImagePixelOutOfBoundsMode(int val) {
		imagePixelOutOfBoundsMode = val;
		}
		
	public boolean getImageGrid() {return imageGrid;}
	public void setImageGrid(boolean val) {
		imageGrid = val;
		}
		
  public Color getImageGridColor() {return imageGridColor;}
  public void setImageGridColor(Color val) {
    imageGridColor = val;
    }
    
	public boolean getCenterImage() {return centerImage;}
	public void setCenterImage(boolean val) {
		centerImage = val;
		}
		
	public boolean getScaleImage() {return scaleImage;}
	public void setScaleImage(boolean val) {
		scaleImage = val;
		}

	public boolean getPreserveImageAspectRatio() {return preserveImageAspectRatio;}
	public void setPreserveImageAspectRatio(boolean val) {
		preserveImageAspectRatio = val;
		}
		
	public int getImageWidth() {return imageWidth;}
	public void setImageWidth(int width) {
		imageWidth = width;
		needsImageCreate = true;
		}
	
	public int getImageHeight() {return imageHeight;}
	public void setImageHeight(int height) {
		imageHeight = height;
		needsImageCreate = true;
		}
		
	public int getImageX() {return imageX;}
	public void setImageX(int val) {
		imageX = val;
		}
		
	public int getImageY() {return imageY;}
	public void setImageY(int val) {
		imageY = val;
		}
		
	public int getImagePixelSize() {return imagePixelSize;}
	public void setImagePixelSize(int val) {
		imagePixelSize = val;
		}
		
	public int getImageTransparentPixel() {return imageTransparentPixel;}
	public void setImageTransparentPixel(int tr) {
		imageTransparentPixel = tr;
		needsImageCreate = true;
		}
		
	public int getImageColorMapSize() {return imageColorMapSize;}
	public void setImageColorMapSize(int c) {
		imageColorMapSize = c;
		needsImageCreate = true;
		}
		
	public int[] getImageColorComponents() {return imageColorComponents;}
	public void setImageColorComponents(int[] c) {
		imageColorComponents = c;
		needsImageCreate = true;
		}
		
  public int getImageGrayLevelMapSize() {return imageGrayLevelMapSize;}
  /** This is a utility function for setting up a graylevel colormap */
  public void setImageGrayLevelMapSize(int c) {
    imageGrayLevelMapSize = c;
    needsImageCreate = true;
    if (imageGrayLevelMapSize > 1) {
      setImageColorMapSize(imageGrayLevelMapSize);
      int[] vals = new int[imageGrayLevelMapSize*3];
      for (int i = 0; i < imageGrayLevelMapSize; ++i) {
        vals[i*3] = vals[i*3 + 1] = vals[i*3 + 2] = 
           255 - (int)(255.0*i/(imageGrayLevelMapSize - 1));
        }
      setImageColorComponents(vals);
      }
    }
    
	public boolean getImageHasAlpha() {return imageHasAlpha;}
	public void setImageHasAlpha(boolean hasAlpha) {
		imageHasAlpha = hasAlpha;
		needsImageCreate = true;
		}
	
  public void setImagePixel(int[] coords, int index) {
    setImagePixel(coords, index, false);
    }
  
  public void setImagePixel(int[] coords, int index, boolean doRepaint) {
    setImagePixel(coords[0], coords[1], index, doRepaint);
    }
    
  public void setImagePixel(int x, int y, int index) {
    setImagePixel(x, y, index, false);
    }
  
  public void setImagePixel(int x, int y, int index, boolean doRepaint) {
    if (needsImageCreate) createImage();
    if (image != null) {
			if (imagePixelOutOfBoundsMode == WRAP) {
				image.getRaster().setPixel(x % imageWidth, y % imageHeight, new int[]{index});
				}
			else {
				if (!(x < 0 || x >= imageWidth || y < 0 || y >= imageHeight))
					image.getRaster().setPixel(x, y, new int[]{index});
				}
      if (doRepaint) repaint();
      }
    }
    
  public int getImagePixel(int x, int y) {
    int[] result = getImagePixel(x, y, null);
    if (result != null) return result[0];
    else return -1;
    }
  
	public int[] getImagePixel(int x, int y, int iArray[]) {
		if (needsImageCreate) createImage();
		if (image != null) {
			return image.getRaster().getPixel(x, y, iArray);
			}
		return null;
		}

	public int[] getImagePixels() {
		return getImagePixels(0, 0, imageWidth, imageHeight, null);
		}
	
	public int[] getImagePixels(int iArray[]) {
		return getImagePixels(0, 0, imageWidth, imageHeight, iArray);
		}
		
	public int[] getImagePixels(int x, int y, int w, int h, int iArray[]) {
		if (needsImageCreate) createImage();
		if (image != null) {
			return image.getRaster().getPixels(x, y, w, h, iArray);
			}
		return null;
		}
	
	public void setImagePixels(int[] pixels) {
		setImagePixels(pixels, true);
		}
		
	public void setImagePixels(int[] pixels, boolean doRepaint) {
    setImagePixelRect(0, 0, imageWidth, imageHeight, pixels, doRepaint);
		}
  
  public void setImagePixels(int[][] pixels) {
    setImagePixels(pixels, true);
    }
    
  public void setImagePixels(int[][] pixels, boolean doRepaint) {
    setImagePixelRect(0, 0, pixels, doRepaint);
    }
  
  
  public void setImagePixelArray(int[][]coords, int[] pixels) {
    setImagePixelArray(coords, pixels, false);
    }
    
  public void setImagePixelArray(int[][]coords, int[] pixels, boolean doRepaint) {
    if (needsImageCreate) createImage();
    if (image != null) {
      int[] onePix = new int[1];
      WritableRaster r = image.getRaster();
      for (int i = 0; i < pixels.length; ++i) {
        onePix[0] = pixels[i];
        int useX = coords[i][0];
        int useY = coords[i][1];
				if (imagePixelOutOfBoundsMode == WRAP) {
					r.setPixel(useX % imageWidth, useY % imageHeight, onePix);
					}
				else {
					if (!(useX < 0 || useX >= imageWidth || useY < 0 || useY >= imageHeight))
						r.setPixel(useX, useY, onePix);
					}
        }
      if (doRepaint) repaint();
      }
    }
    
	public void setImagePixelArray(int x, int y, int[][]offsets, int[] pixels) {
		setImagePixelArray(x, y, offsets, pixels, false);
		}
    
	public void setImagePixelArray(int x, int y, int[][]offsets, int[] pixels, boolean doRepaint) {
		if (needsImageCreate) createImage();
		if (image != null) {
			int[] onePix = new int[1];
			WritableRaster r = image.getRaster();
			for (int i = 0; i < pixels.length; ++i) {
				onePix[0] = pixels[i];
				int useX = x + offsets[i][0];
				int useY = y + offsets[i][1];
				if (imagePixelOutOfBoundsMode == WRAP) {
					r.setPixel(useX % imageWidth, useY % imageHeight, onePix);
					}
				else {
					if (!(useX < 0 || useX >= imageWidth || useY < 0 || useY >= imageHeight))
						r.setPixel(useX, useY, onePix);
					}
				}
			if (doRepaint) repaint();
			}
		}
		
  public void setImagePixelRect(int x, int y, int w, int h, int[] pixels) {
    setImagePixelRect(x, y, w, h, pixels, false);
    }
    
  public void setImagePixelRect(int x, int y, int w, int h, int[] pixels, boolean doRepaint) {
    if (needsImageCreate) createImage();
    if (image != null) {
      image.getRaster().setPixels(x, y, w, h, pixels);
      if (doRepaint) repaint();
      }
    }
  
  public void setImagePixelRect(int x, int y, int[][] pixels) {
    setImagePixelRect(x, y, pixels, false);
    }
    
  public void setImagePixelRect(int x, int y, int[][] pixels, boolean doRepaint) {
    if (needsImageCreate) createImage();
    if (image != null) {
      int w = pixels[0].length;
      int h = pixels.length;
      for (int i = 0; i < h; ++i)
        image.getRaster().setPixels(x, y + i, w, 1, pixels[i]);
      if (doRepaint) repaint();
      }
    }
    
	public void fillImagePixels(int pixel) {
		fillImagePixels(pixel, false);
		}
		
  public void fillImagePixels(int pixel, boolean doRepaint) {
    fillImagePixelRect(0, 0, imageWidth, imageHeight, pixel, doRepaint);
    }
    
  public void fillImagePixelArray(int[][]coords, int pixel) {
    fillImagePixelArray(coords, pixel, false);
    }
    
  public void fillImagePixelArray(int[][]coords, int pixel, boolean doRepaint) {
    if (needsImageCreate) createImage();
    if (image != null) {
      int[] onePix = new int[1];
      WritableRaster r = image.getRaster();
      for (int i = 0; i < coords.length; ++i) {
        onePix[0] = pixel;
        int useX = coords[i][0];
        int useY = coords[i][1];
        if (imagePixelOutOfBoundsMode == WRAP) {
          r.setPixel(useX % imageWidth, useY % imageHeight, onePix);
          }
        else {
          if (!(useX < 0 || useX >= imageWidth || useY < 0 || useY >= imageHeight))
            r.setPixel(useX, useY, onePix);
          }
        }
      if (doRepaint) repaint();
      }
    }
    
  public void fillImagePixelArray(int x, int y, int[][]offsets, int pixel) {
    fillImagePixelArray(x, y, offsets, pixel, false);
    }
    
  public void fillImagePixelArray(int x, int y, int[][]offsets, int pixel, boolean doRepaint) {
    if (needsImageCreate) createImage();
    if (image != null) {
      int[] onePix = new int[1];
      WritableRaster r = image.getRaster();
      for (int i = 0; i < offsets.length; ++i) {
        onePix[0] = pixel;
        int useX = x + offsets[i][0];
        int useY = y + offsets[i][1];
        if (imagePixelOutOfBoundsMode == WRAP) {
          r.setPixel(useX % imageWidth, useY % imageHeight, onePix);
          }
        else {
          if (!(useX < 0 || useX >= imageWidth || useY < 0 || useY >= imageHeight))
            r.setPixel(useX, useY, onePix);
          }
        }
      if (doRepaint) repaint();
      }
    }
    
	public void fillImagePixelRect(int x, int y, int w, int h, int pixel) {
		fillImagePixelRect(x, y, w, h, pixel, false);
		}
    
	public void fillImagePixelRect(int x, int y, int w, int h, int pixel, boolean doRepaint) {
		if (needsImageCreate) createImage();
		if (image != null) {
			int[] a = new int[w*h];
			for (int i=w*h-1; i >= 0; i--)
				a[i] = pixel;
			image.getRaster().setPixels(x, y, w, h, a);
			if (doRepaint) repaint();
			}
		}
		  
  private void createImage() {
    byte[] valBytes = new byte[imageColorComponents.length];
    
    for (int i = 0; i < imageColorComponents.length; ++i) 
      valBytes[i] = (byte)imageColorComponents[i];
      
		IndexColorModel colorModel = new IndexColorModel(8, imageColorMapSize, valBytes, 0, imageHasAlpha, imageTransparentPixel);
		image = new BufferedImage(imageWidth, imageHeight, BufferedImage.TYPE_BYTE_INDEXED, colorModel);
		needsImageCreate = false;
  	}
  	
  private static final int[] NO_PIXEL_LOC = new int[]{-1,-1};
  
	private int useHeight;
	private int useWidth;
	private int compWidth;
	private int compHeight;
	private Insets insets;
	private int left;
	private int top;
	
	// TODO sometime if we know when new computing needs done we 
	// can not always call this but only when needed
	
	private void computeImageInfo() {
		useWidth = imageWidth*imagePixelSize;
		useHeight = imageHeight*imagePixelSize;
		compWidth = getWidth();
		compHeight = getHeight();
		insets = getInsets();
		left = imageX;
		top = imageY;

		if (scaleImage) {
			left = insets.left;
			top = insets.top;
			if (preserveImageAspectRatio) {
				double imRatio = 1.0*useHeight/useWidth;
				if (imRatio >= 1.0) {
					useHeight = compHeight;
					useWidth = (int)(useHeight/imRatio);
					if (useWidth > compWidth) {
						useWidth = compWidth;
						useHeight = (int)(imRatio*useWidth);
						}
					}
				else {
					useWidth = compWidth;
					useHeight = (int)(imRatio*useWidth);
					if (useHeight > compHeight) {
						useHeight = compHeight;
						useWidth = (int)(useHeight/imRatio);
						}
					}
				}
			else {
				useWidth = compWidth;
				useHeight = compHeight;
				}
			}
		if (centerImage) {
			left = (compWidth - useWidth)/2;
			top = (compHeight - useHeight)/2;
			}
		}
	
  public int getImagePixelAt(Point p) {
    int[] coords = getImagePixelCoordinatesAt(p);
    if (coords != null && coords[0] != -1 && coords[1] != -1)
      return getImagePixel(coords[0], coords[1]);
    return -1;
    }
        
  public int getImagePixelAt(MouseEvent event) {
    return getImagePixelAt(event.getPoint());
    }
    
  public int[] getImagePixelCoordinatesAt(Point p) {
		if (imageWidth < 0 || imageHeight < 0) return NO_PIXEL_LOC;
		computeImageInfo();
		if (p.x < left || p.x > (left + useWidth) || p.y < top || p.y > top + useHeight) 
			return NO_PIXEL_LOC;
	  int[] result = new int[]{-1,-1};
	  result[0] = Math.max(0, Math.min((int)(1.0*(p.x-left)*imageWidth/useWidth), imageWidth-1));
		result[1] = Math.max(0, Math.min((int)(1.0*(p.y-top)*imageHeight/useHeight), imageHeight-1));
		return result;
    }
        
	public int[] getImagePixelCoordinatesAt(MouseEvent event) {
		return getImagePixelCoordinatesAt(event.getPoint());
		}
	
	public void repaintNow() {
		paintImmediately(getBounds(null));
		}
	
	public void paintComponent(Graphics g) {
		// We do this so that MacOS X does not antialias the image in the panel
		if (requiresHints) ((Graphics2D)g).addRenderingHints(renderHints);
    
		super.paintComponent(g);
		if (needsImageCreate) createImage();
		if (image != null) paintImage(g);
		}
			
	public void paintImage(Graphics g) {
		if (imageWidth < 0 || imageHeight < 0) return;
		computeImageInfo();
      
		g.drawImage(image, left, top, useWidth, useHeight, this);
		
		// Paint any area outside the image with the background color. Break up border into 4 rects.
		g.clearRect(insets.left, insets.top, compWidth - insets.left - insets.right, top - insets.top);
    g.clearRect(insets.left, top, left - insets.left, useHeight);
    g.clearRect(left + useWidth, top, compWidth - useWidth - left - insets.right, useHeight);
    g.clearRect(insets.left, top + useHeight, compWidth - insets.left - insets.right, compHeight - useHeight - top - insets.bottom);

    if (imageGrid) {
      double xstep = (double)1.0*useWidth/imageWidth;
      double ystep = (double)1.0*useHeight/imageHeight;
      if (xstep >= 2.0 && ystep >= 2.0) {
        // TODO see about allowing a grid line width setting different from 1
        g.setColor(imageGridColor);
        int count = imageWidth;
        double pos = (double)left;
        for (; count >= 0; --count, pos += xstep) {
          g.drawLine((int)pos, top, (int)pos, top + useHeight);
          }
        count = imageHeight;
        pos = (double)top;
        for (; count >= 0; --count, pos += ystep) {
          g.drawLine(left, (int)pos, left + useWidth, (int)pos);
          }
        }
	    }
        
    }
    
  }

/*
 * @(#)TrekToolBar.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.net.URL;

import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JFrame;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JTextField;
import javax.swing.JToggleButton;
import javax.swing.JToolBar;

import com.wolfram.guikit.trek.diva.TrekController;
import com.wolfram.guikit.trek.diva.TrekPane;
import com.wolfram.guikit.trek.diva.TrekFigure;

/**
 * TrekToolBar
 *
 * @version $Revision: 1.3 $
 */
public class TrekToolBar extends JToolBar {
 
    private static final long serialVersionUID = -1287937975436738948L;
  
	protected TrekPane trekPane = null;
	
	protected ButtonGroup modeGroup = new ButtonGroup();
	
    protected JFrame parentFrame;
  
	protected JToggleButton createButton;
    protected JButton selectionDisplayModeButton;
	protected JToggleButton selectButton;
	protected JToggleButton zoomButton;
	protected JToggleButton panButton;
	
    protected JPopupMenu createPopup;
    protected JPopupMenu selectionDisplayModePopup;
    protected JPopupMenu zoomPopup;
  
    protected JTextField colorWell;
  
    protected ImageIcon drawLineIcon;
    protected ImageIcon drawPointsIcon;
  
    protected int selectionDisplayMode = TrekFigure.LINE;
  
  public TrekToolBar() {
    this(null);
    }
    
	public TrekToolBar(TrekPane trekPane) {
		super();
    setFloatable(true);
    setRollover(true);
		setTrekPane(trekPane);
		init();
		}

  public void setSelectionDisplayMode(int displayMode) {
    switch (displayMode) {
      case TrekFigure.POINTS:
        selectionDisplayMode = displayMode;
        selectionDisplayModeButton.setIcon(drawPointsIcon);
        selectionDisplayModeButton.setToolTipText("Set selection to points treks");
        break;
      default:
        selectionDisplayMode = TrekFigure.LINE;
        selectionDisplayModeButton.setIcon(drawLineIcon);
        selectionDisplayModeButton.setToolTipText("Set selection to line treks");
        break;
      }
    }
  
  public JButton getSelectionDisplayModeButton() {return selectionDisplayModeButton;}
  
  public JTextField getColorWell() {return colorWell;}
  
  public JFrame getParentFrame(){return parentFrame;}
  public void setParentFrame(JFrame f) {
    parentFrame = f;
    }
    
	public TrekPane getTrekPane() {return trekPane;}
	public void setTrekPane(TrekPane t) {
		trekPane = t;
    if (trekPane == null) return;
    trekPane.setTrekToolBar(this);
		trekPane.getTrekController().addPropertyChangeListener(TrekController.MODE_PROPERTY,
			new PropertyChangeListener() {
				public void propertyChange(PropertyChangeEvent evt) {
					Object newVal = evt.getNewValue();
					if (newVal != null && newVal instanceof Integer)
						setMode(((Integer)newVal).intValue());
					}
			 	});
		}
		
	public boolean isRequestFocusEnabled() {
		return false;
		}
	
	public void setMode(int mode) {
		switch (mode) {
			case TrekController.MODE_SELECT:
				modeGroup.setSelected(selectButton.getModel(), true);
				break;
			case TrekController.MODE_CREATE:
				modeGroup.setSelected(createButton.getModel(), true);
				break;
			case TrekController.MODE_ZOOM:
				modeGroup.setSelected(zoomButton.getModel(), true);
				break;
			case TrekController.MODE_PAN:
				modeGroup.setSelected(panButton.getModel(), true);
				break;
		 		}
    if (trekPane != null)
		  trekPane.getTrekController().setMode(mode);
		}
	
	public void init() {
    ImageIcon ic = null;
    URL imageURL;
    
    ic = null;
    imageURL = TrekToolBar.class.getClassLoader().getResource("images/trek/select.gif");
    if (imageURL != null) {
      ic = new ImageIcon(imageURL);
      }
		selectButton = new JToggleButton(ic);
    selectButton.setAlignmentX(CENTER_ALIGNMENT);
		selectButton.setSelected(true);
    selectButton.setToolTipText("Change trek selections");
		selectButton.addActionListener(
			new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					setMode(TrekController.MODE_SELECT);
          if (trekPane != null)
					 trekPane.getCanvas().requestFocus();
					}
				}
		  );
      
    drawLineIcon = null;
    imageURL = TrekToolBar.class.getClassLoader().getResource("images/trek/drawLine.gif");
    if (imageURL != null) {
      drawLineIcon = new ImageIcon(imageURL);
      }
    drawPointsIcon = null;
    imageURL = TrekToolBar.class.getClassLoader().getResource("images/trek/drawPoints.gif");
    if (imageURL != null) {
      drawPointsIcon = new ImageIcon(imageURL);
      }
      
		createButton = new JToggleButton(drawLineIcon);
    createButton.setAlignmentX(CENTER_ALIGNMENT);
    createButton.setToolTipText("Create a new line trek");
		createButton.addActionListener(
					new ActionListener() {
						public void actionPerformed(ActionEvent e) {
							setMode(TrekController.MODE_CREATE);
              if (trekPane != null)
							 trekPane.getCanvas().requestFocus();
							}
						}
					);
		
    ic = null;
    imageURL = TrekToolBar.class.getClassLoader().getResource("images/trek/zoomDrop.gif");
    if (imageURL != null) {
      ic = new ImageIcon(imageURL);
      }
    zoomButton = new JToggleButton(ic);
    zoomButton.setAlignmentX(CENTER_ALIGNMENT);
    zoomButton.setToolTipText("Zoom");
    zoomButton.addActionListener(
          new ActionListener() {
            public void actionPerformed(ActionEvent e) {
              setMode(TrekController.MODE_ZOOM);
              if (trekPane != null)
                trekPane.getCanvas().requestFocus();
              }
            }
          );
          
    // Code that supports zoom dropdown in the toolbar
    zoomPopup = new JPopupMenu();
		JMenuItem menuItem = new JMenuItem("Zoom to Fit");
		menuItem.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
					if (trekPane != null) {
						trekPane.zoomToFit();
						trekPane.getCanvas().requestFocus();
						}
			 		}
				});
    zoomPopup.add(menuItem);
    menuItem = new JMenuItem("Scale to Fit");
    menuItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          if (trekPane != null) {
            trekPane.scaleToFit();
            trekPane.getCanvas().requestFocus();
            }
          }
        });
    zoomPopup.add(menuItem);
		zoomButton.addMouseListener(new MouseAdapter() {
			public void mouseClicked(MouseEvent e) {
				if (e.getClickCount() >= 2 && !zoomPopup.isVisible()) {
					zoomPopup.show(zoomButton, 0, zoomButton.getHeight());
					}
				}
			});
    zoomButton.addMouseMotionListener(new MouseMotionAdapter() {
      public void mouseDragged(MouseEvent e) {
        if (!zoomPopup.isVisible()) {
          zoomPopup.show(zoomButton, 0, zoomButton.getHeight());
          }
        }
      });

      
    // Code that supports draw dropdown in the toolbar
    createPopup = new JPopupMenu();
    menuItem = new JMenuItem("Line", drawLineIcon);
    menuItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          if (trekPane != null) {
            trekPane.setDefaultDisplayMode(TrekFigure.LINE);
            trekPane.getCanvas().requestFocus();
            }
          }
        });
    createPopup.add(menuItem);
    menuItem = new JMenuItem("Points", drawPointsIcon);
    menuItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          if (trekPane != null) {
            trekPane.setDefaultDisplayMode(TrekFigure.POINTS);
            trekPane.getCanvas().requestFocus();
            }
          }
        });
    createPopup.add(menuItem);
    createButton.addMouseListener(new MouseAdapter() {
      public void mouseClicked(MouseEvent e) {
        if (e.getClickCount() >= 2 && !createPopup.isVisible()) {
          createPopup.show(createButton, 0, createButton.getHeight());
          }
        }
      });
    createButton.addMouseMotionListener(new MouseMotionAdapter() {
      public void mouseDragged(MouseEvent e) {
        if (!createPopup.isVisible()) {
          createPopup.show(createButton, 0, createButton.getHeight());
          }
        }
      });
      
    ic = null;
    imageURL = TrekToolBar.class.getClassLoader().getResource("images/trek/pan.gif");
    if (imageURL != null) {
      ic = new ImageIcon(imageURL);
      }
		panButton = new JToggleButton(ic);
    panButton.setAlignmentX(CENTER_ALIGNMENT);
    panButton.setToolTipText("Pan");
		panButton.addActionListener(
					new ActionListener() {
						public void actionPerformed(ActionEvent e) {
							setMode(TrekController.MODE_PAN);
              if (trekPane != null)
							 trekPane.getCanvas().requestFocus();
							}
						}
					);

    colorWell = new JTextField() {
      private static final long serialVersionUID = -1287187975416781948L;
      public boolean isFocusTraversable() {return false;}
       };
    Dimension wellSize = new Dimension(20,20);
    colorWell.setPreferredSize(wellSize);
    colorWell.setMinimumSize(wellSize);
    colorWell.setMaximumSize(wellSize);
    colorWell.setSize(20,20);
    colorWell.setEditable(false);
    colorWell.setBackground(Color.WHITE);
    colorWell.setToolTipText("Click to edit trek color");
    colorWell.addMouseListener( new MouseAdapter() {
      public void mousePressed(MouseEvent e) {
        if (!colorWell.isEnabled()) return;
        Color newColor = JColorChooser.showDialog(
          parentFrame,
          "Choose Trek Color",
          colorWell.getBackground());
        if (newColor != null) {
          colorWell.setBackground(newColor);
          setSelectionColor(newColor);
          }
        }
      });
      
    selectionDisplayModeButton = new JButton(drawLineIcon);
    selectionDisplayModeButton.setAlignmentX(CENTER_ALIGNMENT);
    selectionDisplayModeButton.setToolTipText("Set selection to line treks");
    selectionDisplayModeButton.addActionListener(
          new ActionListener() {
            public void actionPerformed(ActionEvent e) {
              updateSelectionDisplayMode();
              if (trekPane != null)
               trekPane.getCanvas().requestFocus();
              }
            }
          );
    // Code that supports draw dropdown in the toolbar
    selectionDisplayModePopup = new JPopupMenu();
    menuItem = new JMenuItem("Line", drawLineIcon);
    menuItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          setSelectionDisplayMode(TrekFigure.LINE);
          updateSelectionDisplayMode();
          if (trekPane != null) {
            trekPane.getCanvas().requestFocus();
            }
          }
        });
    selectionDisplayModePopup.add(menuItem);
    menuItem = new JMenuItem("Points", drawPointsIcon);
    menuItem.addActionListener(new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          setSelectionDisplayMode(TrekFigure.POINTS);
          updateSelectionDisplayMode();
          if (trekPane != null) {
            trekPane.getCanvas().requestFocus();
            }
          }
        });
    selectionDisplayModePopup.add(menuItem);
    selectionDisplayModeButton.addMouseListener(new MouseAdapter() {
      public void mouseClicked(MouseEvent e) {
        if (e.getClickCount() >= 2 && !selectionDisplayModePopup.isVisible()) {
          selectionDisplayModePopup.show(selectionDisplayModeButton, 0, selectionDisplayModeButton.getHeight());
          }
        }
      });
    selectionDisplayModeButton.addMouseMotionListener(new MouseMotionAdapter() {
      public void mouseDragged(MouseEvent e) {
        if (!selectionDisplayModePopup.isVisible()) {
          selectionDisplayModePopup.show(selectionDisplayModeButton, 0, selectionDisplayModeButton.getHeight());
          }
        }
      });
      
		modeGroup.add(selectButton);
		modeGroup.add(createButton);
		modeGroup.add(zoomButton);
		modeGroup.add(panButton);
		
		add(selectButton);
		add(createButton);
		add(zoomButton);
		add(panButton);
    
    addSeparator();
    add(Box.createRigidArea(new Dimension(3,3)));
    add(colorWell);
    add(Box.createRigidArea(new Dimension(2,2)));
    add(selectionDisplayModeButton);
    
		}
	
  public void setDefaultDisplayMode(int mode) {
    switch (mode) {
      case TrekFigure.LINE:
        createButton.setIcon(drawLineIcon);
        createButton.setToolTipText("Create a new line trek");
        break;
      case TrekFigure.POINTS:
        createButton.setIcon(drawPointsIcon);
        createButton.setToolTipText("Create a new points trek");
        break;
        } 
    }
  
  protected void setSelectionColor(Color newColor) {
    if (trekPane != null)
      trekPane.setSelectionColor(newColor);
    }
    
  protected void updateSelectionDisplayMode() {
    if (trekPane != null)
      trekPane.setSelectionDisplayMode(selectionDisplayMode);
    }
    
}
/*
 * @(#)JFontChooser.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.*;
import java.awt.event.*;
import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.Vector;

import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import javax.swing.event.*;
import javax.swing.*;

/**
 * A reusable Font Chooser dialog that implements similar functionality to
 * the JColorChooser component provided with Swing.<p>
 * Upon initialization, the JFontChooser polls the system for all available
 * fonts, initializes the various JList components to the values of the 
 * default font and provides a preview of the font. As options are changed/selected
 * the preview is updated to display the current font.<p>
 * JFontChooser can either be created and added to a GUI as a typical
 * JComponent or it can display a JDialog using the {@link #showDialog(Component, String) showDialog}
 * method (just like the <b>JColorChooser</b> component).<p>
 * <p>
 */
public class JFontChooser extends JComponent implements ActionListener, ListSelectionListener {

    private static final long serialVersionUID = -1285987575451788948L;
    
    private FontSelectionModel selectionModel;

    /**
    * The selection model property name.
    */
    public static final String SELECTION_MODEL_PROPERTY = "selectionModel";

    /**
    * The preview panel property name.
    */
    public static final String PREVIEW_PANEL_PROPERTY = "previewPanel";
    
    private static final String[] LOGICAL_NAMES = new String[]{ "Default", "Dialog", "DialogInput", 
          "Serif", "SansSerif", "Monospaced"};
          
	private boolean dragEnabled;
  
    private boolean allowsAnySize = true;
    private boolean showLogicalFonts = true;
    private boolean showPhysicalFonts = true;
    private String[] customFontNames = null;
  
    private JList fontNames, fontSizes, fontStyles;
	private JTextField currentSize;
  
	private JComponent previewPanel;
    private JSplitPane splitPane;
	/**
	 * Value returned by {@link #showDialog(Component, String) showDialog} upon an error.
	 */
	public static final int ERROR_OPTION=0;
	/**
	 * Value returned by {@link #showDialog(Component, String) showDialog} upon a selected font.
	 */
	public static final int ACCEPT_OPTION=2;
	/**
	 * Value returned by {@link #showDialog(Component, String) showDialog} upon a cancel.
	 */
	public static final int CANCEL_OPTION=4;
	
  public static final String STYLE_PLAIN_NAME = "Plain";
  public static final String STYLE_BOLD_NAME = "Bold";
  public static final String STYLE_ITALIC_NAME = "Italic";
  public static final String STYLE_BOLDITALIC_NAME = "Bold-Italic";
  
  public static final Font DEFAULT_FONT = new Font("Serif", Font.PLAIN, 12);
  
  public JFontChooser() {
    this(DEFAULT_FONT);
    }
  
	/**
	 * Constructs a new JFontChooser component initialized to the supplied font object.
	 * @see	JFontChooser#showDialog(Component, String)
	 */
	public JFontChooser(Font initialFont) {
    this(new DefaultFontSelectionModel(initialFont));
    }
	
  public JFontChooser(FontSelectionModel model) {
    super();
    selectionModel = model;
    setup();
    dragEnabled = false;
    }
    
  /**
   * Shows a modal color-chooser dialog and blocks until the
   * dialog is hidden.  If the user presses the "OK" button, then
   * this method hides/disposes the dialog and returns the selected color.
   * If the user presses the "Cancel" button or closes the dialog without
   * pressing "OK", then this method hides/disposes the dialog and returns
   * <code>null</code>.
   *
   * @param component    the parent <code>Component</code> for the dialog
   * @param title        the String containing the dialog's title
   * @param initialColor the initial Color set when the color-chooser is shown
   * @return the selected color or <code>null</code> if the user opted out
   * returns true.
   * @see java.awt.GraphicsEnvironment#isHeadless
   */
  public static Font showDialog(Component component,
    String title, Font initialFont) {

    final JFontChooser pane = new JFontChooser(initialFont != null? initialFont : DEFAULT_FONT);

    FontTracker ok = new FontTracker(pane);
    JDialog dialog = createDialog(component, title, true, pane, ok, null);
    dialog.addWindowListener(new FontChooserDialog.Closer());
    dialog.addComponentListener(new FontChooserDialog.DisposeOnClose());
    dialog.show(); // blocks until user brings dialog down...
    return ok.getFont();
    }

  /**
   * Creates and returns a new dialog containing the specified
   * <code>ColorChooser</code> pane along with "OK", "Cancel", and "Reset"
   * buttons. If the "OK" or "Cancel" buttons are pressed, the dialog is
   * automatically hidden (but not disposed).  If the "Reset"
   * button is pressed, the color-chooser's color will be reset to the
   * color which was set the last time <code>show</code> was invoked on the
   * dialog and the dialog will remain showing.
   *
   * @param c              the parent component for the dialog
   * @param title          the title for the dialog
   * @param modal          a boolean. When true, the remainder of the program
   *                       is inactive until the dialog is closed.
   * @param chooserPane    the color-chooser to be placed inside the dialog
   * @param okListener     the ActionListener invoked when "OK" is pressed
   * @param cancelListener the ActionListener invoked when "Cancel" is pressed
   * @return a new dialog containing the color-chooser pane
   * returns true.
   * @see java.awt.GraphicsEnvironment#isHeadless
   */
  public static JDialog createDialog(Component c, String title, boolean modal,
    JFontChooser chooserPane, ActionListener okListener,
    ActionListener cancelListener) {

    return new FontChooserDialog(c, title, modal, chooserPane, okListener, cancelListener);
    }
    
	private void setup() {
		setLayout(new BorderLayout());
		
		String[] fontList = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();

		fontNames = new JList(fontList);
		fontNames.setVisibleRowCount(10);
		JScrollPane fontNamesScroll = new JScrollPane(fontNames);
		fontNames.addListSelectionListener(this);
    JPanel fontNamesPane = new JPanel(new BorderLayout());
		fontNamesPane.setBorder(new EmptyBorder(2,2,2,2));
		fontNamesPane.add(new JLabel("Font Family:"), BorderLayout.NORTH);
		fontNamesPane.add(fontNamesScroll, BorderLayout.CENTER);
		
		Object[] styles = {STYLE_PLAIN_NAME, STYLE_BOLD_NAME, STYLE_ITALIC_NAME, STYLE_BOLDITALIC_NAME};
		fontStyles = new JList(styles);
		JScrollPane fontStylesScroll = new JScrollPane(fontStyles);
		fontStyles.setSelectedIndex(0);
		fontStyles.addListSelectionListener(this);
		JPanel fontStylesPane = new JPanel(new BorderLayout());
		fontStylesPane.setBorder(new EmptyBorder(2,0,2,0));
		fontStylesPane.add(new JLabel("Style:"), BorderLayout.NORTH);
		fontStylesPane.add(fontStylesScroll, BorderLayout.CENTER);
		
		String[] sizes = new String[] {"8","9","10","11","12","14","16","18","20","24","28","32","36","48","72"};
		fontSizes = new JList(sizes);
		fontSizes.setVisibleRowCount(8);
		JScrollPane fontSizesScroll = new JScrollPane(fontSizes);
		fontSizes.addListSelectionListener(this);
		
		currentSize = new JTextField(5);
		currentSize.addActionListener(this);
		
		JPanel sizePane = new JPanel(new BorderLayout());
		sizePane.add(new JLabel("Size:"), BorderLayout.NORTH);
		sizePane.add(currentSize, BorderLayout.SOUTH);
		
		JPanel fontSizesPane = new JPanel(new BorderLayout());
		fontSizesPane.setBorder(new EmptyBorder(2,2,2,2));
		fontSizesPane.add(sizePane, BorderLayout.NORTH);
		fontSizesPane.add(fontSizesScroll, BorderLayout.CENTER);
				
		previewPanel = new DefaultPreviewPanel();
		
		JPanel top = new JPanel(new BorderLayout());
		top.add(fontNamesPane, BorderLayout.CENTER);

		JPanel topSide = new JPanel(new BorderLayout());
		topSide.add(fontStylesPane, BorderLayout.WEST);
		topSide.add(fontSizesPane, BorderLayout.EAST);
		top.add(topSide, BorderLayout.EAST);
		
		splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
		splitPane.setTopComponent(top);
		splitPane.setBottomComponent(previewPanel);
		add(splitPane, BorderLayout.CENTER);
		
    updateFamilyScroll();
    updateStyleScroll();
    updateSizeField();
    updateSizeScroll();
	}
	
  public boolean getAllowAnySize() {return allowsAnySize;}
  public void setAllowAnySize(boolean val) {allowsAnySize = val;}
  
  public boolean getShowLogicalFonts() {return showLogicalFonts;}
  public void setShowLogicalFonts(boolean val) {
    showLogicalFonts = val;
    updateFontNames(customFontNames);
    }
  public boolean getShowPhysicalFonts() {return showPhysicalFonts;}
  public void setShowPhysicalFonts(boolean val) {
    showPhysicalFonts = val;
    updateFontNames(customFontNames);
    }
  
  private void updateFontNames(String[] names) {
    Vector newData = new Vector();
    if (names == null) {
      if (showPhysicalFonts) {
        String[] fontList = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
        for (int i = 0; i < fontList.length; ++i) {
          newData.add(fontList[i]);
          }
        if (!showLogicalFonts) {
          for (int i = 0; i < LOGICAL_NAMES.length; ++i) {
            newData.remove(LOGICAL_NAMES[i]);
            }
          }
        }
      else if (showLogicalFonts) {
        for (int i = 0; i < LOGICAL_NAMES.length; ++i) {
          newData.add(LOGICAL_NAMES[i]);
          }
        }
      }
    else {
      for (int i = 0; i < names.length; ++i) {
        newData.add(names[i]);
        }
      }
    
    Collections.sort(newData);
    fontNames.setListData(newData);
    // Need to validate current font name choice if not present
    if (getFont() != null) {
			String useName = getFont().getFamily();
			String newName = useName;
			
			ListModel model = fontNames.getModel();
			for (int i = model.getSize()-1; i >=0; --i) {
				newName = (String)model.getElementAt(i);
				if (newName.equals(useName))
					break;
				}
								
			if(!newName.equals(useName)) {
				int useStyle = getFont().getStyle();
				int useSize = getFont().getSize();
				selectionModel.setSelectedFont(new Font(newName, useStyle, useSize));
				updatePreview();
				}
      updateFamilyScroll();
      }
    }
  
  public void setFamilyNames(String[] names) {
    customFontNames = names;
    updateFontNames(names);
    }
    
  public void setSizes(String[] sizes) {
    Vector newData = new Vector(sizes.length);
    for (int i = 0; i < sizes.length; ++i)
      newData.add(sizes[i]);
    Collections.sort(newData,
      new Comparator() {
        public int compare(Object o1, Object o2) {
          return new Integer((String)o1).compareTo( new Integer((String)o2));
          }
        public boolean equals(Object obj) {
          return this.equals(obj);
          }
       });
    fontSizes.setListData(newData);
    // Might need to validate current size value
    if (getFont() != null)
      setValidSize(getFont().getSize());
    }
    
	public void setFont(Font f) {
    selectionModel.setSelectedFont(f);
    updateFamilyScroll();
    updateStyleScroll();
    updateSizeField();
    updateSizeScroll();
    updatePreview();
    }
	
  public Font getFont() {
    return selectionModel.getSelectedFont();
    }

    /**
     * Sets the <code>dragEnabled</code> property,
     * which must be <code>true</code> to enable
     * automatic drag handling (the first part of drag and drop)
     * on this component.
     * The <code>transferHandler</code> property needs to be set
     * to a non-<code>null</code> value for the drag to do
     * anything.  The default value of the <code>dragEnabled</code>
     * property
     * is <code>false</code>.
     *
     * <p>
     *
     * When automatic drag handling is enabled,
     * most look and feels begin a drag-and-drop operation
     * when the user presses the mouse button over the preview panel.
     * Some look and feels might not support automatic drag and drop;
     * they will ignore this property.  You can work around such
     * look and feels by modifying the component
     * to directly call the <code>exportAsDrag</code> method of a
     * <code>TransferHandler</code>.
     *
     * @param b the value to set the <code>dragEnabled</code> property to
     *
     * @since 1.4
     *
     * @see java.awt.GraphicsEnvironment#isHeadless
     * @see #getDragEnabled
     * @see #setTransferHandler
     * @see TransferHandler
     *
     * @beaninfo
     *  description: Determines whether automatic drag handling is enabled.
     *        bound: false
     */
    public void setDragEnabled(boolean b) {
      dragEnabled = b;
    }

    /**
     * Gets the value of the <code>dragEnabled</code> property.
     *
     * @return  the value of the <code>dragEnabled</code> property
     * @see #setDragEnabled
     * @since 1.4
     */
    public boolean getDragEnabled() {
      return dragEnabled;
    }
    
  /**
   * Returns the data model that handles font selections.
   *
   * @return a <code>FontSelectionModel</code> object 
   */
  public FontSelectionModel getSelectionModel() {
    return selectionModel;
    }


  /**
   * Sets the model containing the selected font.
   *
   * @param newModel   the new <code>FontSelectionModel</code> object
   *
   * @beaninfo
   *       bound: true
   *      hidden: true
   * description: The model which contains the currently selected font.
   */
  public void setSelectionModel(FontSelectionModel newModel ) {
    FontSelectionModel oldModel = selectionModel;
    selectionModel = newModel;
    firePropertyChange(JFontChooser.SELECTION_MODEL_PROPERTY, oldModel, newModel);
    }
    
  /**
   * Sets the current preview panel.
   * This will fire a <code>PropertyChangeEvent</code> for the property
   * named "previewPanel".
   *
   * @param preview the <code>JComponent</code> which displays the current font
   * @see JComponent#addPropertyChangeListener
   *
   * @beaninfo
   *       bound: true
   *      hidden: true
   * description: The UI component which displays the current font.
   */
  public void setPreviewPanel(JComponent preview) {
    if (previewPanel != preview) {
      JComponent oldPreview = previewPanel;
      previewPanel = preview;
      // If this is Null do we need to force repaint or change slider position to full?
      // Probably, need to test
      if (previewPanel != null)
        splitPane.setBottomComponent(null);
      splitPane.setBottomComponent(previewPanel);
      updatePreview();
      firePropertyChange(JColorChooser.PREVIEW_PANEL_PROPERTY, oldPreview, preview);
      }
    }

  /**
   * Returns the preview panel that shows a chosen font.
   *
   * @return a <code>JComponent</code> object -- the preview panel
   */
  public JComponent getPreviewPanel() {
    return previewPanel;
    }
    
  private void updatePreview() {
    if (previewPanel != null)
      previewPanel.setFont(getFont());
    }
  
  private void updateSizeField() {
    if (getFont() != null) {
      currentSize.setText("" + getFont().getSize());
      }
    }
  private void updateSizeScroll() {
    if (getFont() != null) {
      fontSizes.setSelectedValue("" + getFont().getSize(), true);
      }
    }
  private void updateStyleScroll() {
    if (getFont() != null) {
      if (getFont().getStyle() == Font.PLAIN)
        fontStyles.setSelectedValue(STYLE_PLAIN_NAME, false);
      else if (getFont().getStyle() == Font.ITALIC)
        fontStyles.setSelectedValue(STYLE_ITALIC_NAME, false);
      else if (getFont().getStyle() == Font.BOLD)
        fontStyles.setSelectedValue(STYLE_BOLD_NAME, false);
      else if (getFont().getStyle() == (Font.BOLD | Font.ITALIC))
        fontStyles.setSelectedValue(STYLE_BOLDITALIC_NAME, false);
      }
    }
  
  private void updateFamilyScroll() {
    if (getFont() != null) {
      fontNames.setSelectedValue("" + getFont().getFamily(), true);
      }
    }
    
  private void setValidSize(int useSize) {
    if (getFont() != null) {
      if (!allowsAnySize) {
        ListModel model = fontSizes.getModel();
        String valString = "" + useSize;
        String currString = valString;
        for (int i = model.getSize()-1; i >=0; --i) {
          currString = (String)model.getElementAt(i);
          if (valString.equals(currString))
            break;
          }
        useSize = Integer.parseInt(currString);
        }
      selectionModel.setSelectedFont(getFont().deriveFont((new Integer(useSize)).floatValue()));
      updatePreview();
      updateSizeField();
      updateSizeScroll();
      }
    }
  
	public void actionPerformed(ActionEvent e) {
		if (e.getSource() == currentSize) {
      int useSize = 12;
      try {
        useSize = Integer.parseInt(currentSize.getText());
        } 
      catch (Exception ex) {}
      setValidSize(useSize);
		  }
    }
	
	/**
	 * Processes events received from the various JList objects.
	 */
	public void valueChanged(ListSelectionEvent e) {
    int useStyle = Font.PLAIN;
    int useSize = 12;
    String useName = DEFAULT_FONT.getFamily();
    
    if (e.getValueIsAdjusting()) return;
    
		if (e.getSource() == fontNames) {
      if (fontNames.getSelectedValue() != null)
        useName = (String)fontNames.getSelectedValue();
      if (getFont() != null) {
        useStyle = getFont().getStyle();
        useSize = getFont().getSize();
        selectionModel.setSelectedFont(new Font(useName, useStyle, useSize));
        updatePreview();
        }
      else {
        setFont(new Font(useName, useStyle, useSize));
        }
		  }
		else if (e.getSource() == fontSizes) {
      if (fontSizes.getSelectedValue() != null) {
        try {
          useSize = Integer.parseInt((String)fontSizes.getSelectedValue());
          } 
        catch (Exception ex) {}
        if (getFont() != null) {
          selectionModel.setSelectedFont(getFont().deriveFont((new Integer(useSize)).floatValue()));
          updateSizeField();
          updatePreview();
          }
        }
		  }
		else if (e.getSource() == fontStyles) {
      if (fontStyles.getSelectedValue() != null) {
  			String selection = (String)fontStyles.getSelectedValue();
  			if (selection.equals(STYLE_PLAIN_NAME))
  				useStyle = Font.PLAIN;
  			else if (selection.equals(STYLE_BOLD_NAME))
  				useStyle = Font.BOLD;
  			else if (selection.equals(STYLE_ITALIC_NAME))
  				useStyle = Font.ITALIC;
  			else if (selection.equals(STYLE_BOLDITALIC_NAME))
  				useStyle = (Font.BOLD | Font.ITALIC);
        if (getFont() != null) {
          selectionModel.setSelectedFont(getFont().deriveFont(useStyle));
          updatePreview();
          }
        }
		  }
	}
	
  // exists for testing purposes only
  public static void main(String args[]) {
    try {
      UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
    }
    catch (Exception e){}
    
    //JFontChooser thisChooser = new JFontChooser();
    //thisChooser.setAllowAnySize(true);
    //thisChooser.setShowLogicalFonts(true);
	  //thisChooser.setShowPhysicalFonts(true);
    //thisChooser.setFamilyNames(new String[]{"Dialog", "Courier New"});
    //JFrame frame = new JFrame();
    //frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    //frame.getContentPane().add(thisChooser);
    //frame.pack();
    //frame.show();

    Font chosenFont = JFontChooser.showDialog(null, "Font choice test", null);
    System.out.println("Font :" + chosenFont.toString());
    }
  
  class DefaultPreviewPanel extends JLabel {
    private static final long serialVersionUID = -1282987975456782942L;
      
    public DefaultPreviewPanel() {
      super("None");
      setHorizontalAlignment(CENTER);
      setVerticalAlignment(CENTER);
      setBorder(new TitledBorder("Sample"));
      }
    public void setFont(Font f) {
      if (f != null) {
        super.setFont(f);
        setText(f.getFamily());
        }
      else {
        super.setFont(DEFAULT_FONT);
        setText("No font selected");
        }
      }
    public Dimension getPreferredSize() {return new Dimension(getSize().width, 75);}
    public Dimension getMinimumSize() {return getPreferredSize();}
  }

}
		
/*
 * Class which builds a color chooser dialog consisting of
 * a JColorChooser with "Ok", "Cancel", and "Reset" buttons.
 *
 * Note: This needs to be fixed to deal with localization!
 */
class FontChooserDialog extends JDialog {
  
    private static final long serialVersionUID = -1237987975356388948L;
    
    private Font initialFont;
    private JFontChooser chooserPane;

    public FontChooserDialog(Component c, String title, boolean modal,
        JFontChooser chooserPane, ActionListener okListener, ActionListener cancelListener) {
			super(JOptionPane.getFrameForComponent(c), title, modal);
			//setResizable(false);

			this.chooserPane = chooserPane;

  		String okString = UIManager.getString("ColorChooser.okText");
  		String cancelString = UIManager.getString("ColorChooser.cancelText");
  		String resetString = UIManager.getString("ColorChooser.resetText");

			Container contentPane = getContentPane();
			contentPane.setLayout(new BorderLayout());
 			contentPane.add(chooserPane, BorderLayout.CENTER);

        /*
         * Create Lower button panel
         */
  		JPanel buttonPane = new JPanel();
			buttonPane.setLayout(new FlowLayout(FlowLayout.CENTER));
			JButton okButton = new JButton(okString);
  		getRootPane().setDefaultButton(okButton);
			okButton.setActionCommand("OK");
			if (okListener != null) {
				okButton.addActionListener(okListener);
				}
 			okButton.addActionListener(new ActionListener() {
				public void actionPerformed(ActionEvent e) {
     			hide();
					}
        });
			buttonPane.add(okButton);

			final JButton cancelButton = new JButton(cancelString);

  		// The following few lines are used to register esc to close the dialog
  		Action cancelKeyAction = new AbstractAction() {
          private static final long serialVersionUID = -1237987975453738948L;
  		  public void actionPerformed(ActionEvent e) {
  		      ActionListener[] listeners = cancelButton.getActionListeners();
  		        for (int i = 0; i < listeners.length; ++i)
  		            listeners[i].actionPerformed(e);
  		      }
  		  }; 
  		
  		KeyStroke cancelKeyStroke = KeyStroke.getKeyStroke(KeyEvent.VK_ESCAPE,0);
  		InputMap inputMap = cancelButton.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW);
  		ActionMap actionMap = cancelButton.getActionMap();
  		if (inputMap != null && actionMap != null) {
      	inputMap.put(cancelKeyStroke, "cancel");
      	actionMap.put("cancel", cancelKeyAction);
  			}
  		// end esc handling

			cancelButton.setActionCommand("cancel");
			if (cancelListener != null) {
 				cancelButton.addActionListener(cancelListener);
        }
			cancelButton.addActionListener(new ActionListener() {
 				public void actionPerformed(ActionEvent e) {
     			hide();
					}
        });
			buttonPane.add(cancelButton);

			JButton resetButton = new JButton(resetString);
			resetButton.addActionListener(new ActionListener() {
  			public void actionPerformed(ActionEvent e) {
   				reset();
   				}
        });
			int mnemonic = UIManager.getInt("ColorChooser.resetMnemonic");
 			if (mnemonic != -1) {
    		resetButton.setMnemonic(mnemonic);
        }
      buttonPane.add(resetButton);
			contentPane.add(buttonPane, BorderLayout.SOUTH);

			if (JDialog.isDefaultLookAndFeelDecorated()) {
   			boolean supportsWindowDecorations = 
   					UIManager.getLookAndFeel().getSupportsWindowDecorations();
   			if (supportsWindowDecorations) {
    			getRootPane().setWindowDecorationStyle(JRootPane.COLOR_CHOOSER_DIALOG);
    			}
        }
 			applyComponentOrientation(((c == null) ? getRootPane() : c).getComponentOrientation());

  		pack();
			setLocationRelativeTo(c);
    	}

    public void show() {
			initialFont = chooserPane.getFont();
			super.show();
    	}

    public void reset() {
 			chooserPane.setFont(initialFont);
    	}

    static class Closer extends WindowAdapter implements Serializable{
        private static final long serialVersionUID = -1287987575455785948L;
        public void windowClosing(WindowEvent e) {
     		Window w = e.getWindow();
    		w.hide();
          }
    	}

    static class DisposeOnClose extends ComponentAdapter implements Serializable{
        private static final long serialVersionUID = -1287927975456728928L;
  		public void componentHidden(ComponentEvent e) {
     		Window w = (Window)e.getComponent();
     		w.dispose();
          }
    	}

	}

  class FontTracker implements ActionListener, Serializable {
      
    private static final long serialVersionUID = -1283987975456388648L;
      
    JFontChooser chooser;
    Font font;

    public FontTracker(JFontChooser c) {
			chooser = c;
    	}

    public void actionPerformed(ActionEvent e) {
 		 font = chooser.getFont();
    	}

    public Font getFont() {
			return font;
    	}
}
		
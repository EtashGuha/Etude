/*
 * @(#)SystemPropertiesJPanel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.datatransfer.Clipboard;
import java.awt.datatransfer.StringSelection;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.Iterator;
import java.util.Properties;
import java.util.TreeMap;

import javax.swing.AbstractAction;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JScrollPane;
import javax.swing.JSeparator;
import javax.swing.KeyStroke;
import javax.swing.table.DefaultTableColumnModel;
import javax.swing.table.TableColumn;
import javax.swing.table.TableColumnModel;

import com.wolfram.bsf.engines.MathematicaBSFEngine;
import com.wolfram.guikit.type.ExprAccessible;
import com.wolfram.guikit.type.ItemAccessible;
import com.wolfram.guikit.swing.table.SortTableModel;
import com.wolfram.jlink.Expr;

/**
 * SystemPropertiesJPanel is a utility subclass of JPanel 
 * that displays a table of current system properties defined
 * with a contextual popmenu supporting copy of rows and a refresh of the data
 */
public class SystemPropertiesJPanel extends JPanel implements ExprAccessible, ItemAccessible {
  
  private static final long serialVersionUID = -1287985975456785941L;
    
  private static final String COLUMN_PROP = "Property";
  private static final String COLUMN_VALUE = "Value";
    
  private static final String BORDER_TITLE = "System Properties";
    
  private static final int TOOLTIP_LINEBREAK = 120;
  
	protected SystemPropertiesJTable table;
	protected SystemPropertiesTableModel dataModel;
    
	protected JPopupMenu popup;
	
  public SystemPropertiesJPanel() {
    super();
    createPropertiesPanel();
    }

  private String multilineHTMLTooltip(String str, int len) {
  	if (str == null) return null;
  	int strLen = str.length();
  	if (strLen <= len) return str;
  	StringBuffer buff = new StringBuffer(str);
  	buff.insert(0, "<html>");
  	int count = strLen/len;
  	for (int i = 1; i < count; ++i)
  		buff.insert(6 + i*len + i*4, "<br>");
		buff.append("</html>");
    return buff.toString();
  	}
  
  private String join(String[] eles, String sep) {
  	 StringBuffer buff = new StringBuffer("");
  	 if (eles == null || eles.length == 0) return "";
  	 if (eles.length == 1) return eles[0];
  	 buff.append(eles[0]);
  	 for (int i = 1; i < eles.length; ++i) {
  	 		buff.append(sep);
  	 		buff.append(eles[i]);
  	 		}
  	 return buff.toString();
 		 }
  
  public void refresh() {
    createTableModel();
    table.setModel(dataModel);
    }
  
  protected void createTableModel() {
    // Create the data model
    Properties props = System.getProperties();
    Iterator keys = props.keySet().iterator();
    String key;
  
    // Store key values in a TreeMap which will sort the
    // keys alphabetically.
    TreeMap properties = new TreeMap();

    while (keys.hasNext()) {
      key = (String)keys.next();
      properties.put(key, props.getProperty(key));
      }
    
    try {
    properties.put("wolfram.jlink.system.id",
      join(com.wolfram.jlink.Utils.getSystemID(), ","));
    
    properties.put("wolfram.jlink.version", com.wolfram.jlink.Utils.getJLinkVersion());
    properties.put("wolfram.jlink.package.context", com.wolfram.jlink.KernelLink.PACKAGE_CONTEXT);
    properties.put("wolfram.jlink.raggedarrays", "" + com.wolfram.jlink.Utils.isRaggedArrays());
    properties.put("wolfram.jlink.jar.dir", com.wolfram.jlink.Utils.getJLinkJarDir());

    properties.put("wolfram.jlink.class.path", 
       join( MathematicaBSFEngine.getClassLoaderHandler().getClassPath(), System.getProperty("path.separator")));
      
    properties.put("wolfram.guikit.version", com.wolfram.guikit.GUIKitEnvironment.VERSION);
    properties.put("wolfram.guikit.version.number", "" + com.wolfram.guikit.GUIKitEnvironment.VERSION_NUMBER);
    properties.put("wolfram.guikit.package.context", com.wolfram.guikit.GUIKitEnvironment.PACKAGE_CONTEXT);
    }
    catch (Exception ex) {ex.printStackTrace();}
    
    dataModel = new SystemPropertiesTableModel(properties);
    }
  
  protected void createPropertiesPanel()  {
    createTableModel();

    DefaultTableColumnModel columnModel = new DefaultTableColumnModel();
    TableColumn column = new TableColumn();
    column.setHeaderValue(COLUMN_PROP);
    column.setPreferredWidth(150);
    column.setMinWidth(25);
    columnModel.addColumn(column);

    column = new TableColumn(1);
    column.setHeaderValue(COLUMN_VALUE);
    column.setPreferredWidth(200);
    columnModel.addColumn(column);
    
		popup = new JPopupMenu();
		
    Action copySelectedAction = new AbstractAction("copy.selected") {
        private static final long serialVersionUID = -1287987272456788942L;
        public void actionPerformed(ActionEvent evt) {
          copySelectedRows();
          }
        };
        
		JMenuItem menuItem = new JMenuItem("Copy All Rows");
		Font popUpFont = new Font(menuItem.getFont().getName(),
			Font.PLAIN, menuItem.getFont().getSize());
		menuItem.setFont( popUpFont);
		menuItem.addActionListener(new AbstractAction("copy.all") {
          private static final long serialVersionUID = -1287917975156718948L;
          public void actionPerformed(ActionEvent evt) {
            copyAllRows();
            }
          });
		popup.add(menuItem);

		menuItem = new JMenuItem("Copy Selected Rows");
		menuItem.setFont( popUpFont);
		menuItem.addActionListener(copySelectedAction);
		popup.add(menuItem);
		
    popup.add(new JSeparator());
    
    menuItem = new JMenuItem("Refresh");
    menuItem.setFont( popUpFont);
    menuItem.addActionListener(new AbstractAction("refresh") {
      private static final long serialVersionUID = -1282987975226782948L;
      public void actionPerformed(ActionEvent evt) {
        refresh();
        }
      });
    popup.add(menuItem);
    
    // Put the table and the UI together.
    table = new SystemPropertiesJTable(dataModel, columnModel);
    JScrollPane pane = new JScrollPane(table);
    pane.setPreferredSize(new Dimension(350, 200));

		MouseListener popupListener = new PopupListener();
		table.addMouseListener(popupListener);
      
    table.getInputMap().put(
      KeyStroke.getKeyStroke(KeyEvent.VK_C, Toolkit.getDefaultToolkit().getMenuShortcutKeyMask()), 
      copySelectedAction.getValue(Action.NAME));
    table.getActionMap().put(copySelectedAction.getValue(Action.NAME), copySelectedAction);
		
    setLayout(new BorderLayout());
    add(pane, BorderLayout.CENTER);
    setBorder(BorderFactory.createTitledBorder(BORDER_TITLE));
    }
    
  public void copyAllRows() {
		Clipboard system;
		StringSelection stsel;
		
		StringBuffer sbf = new StringBuffer();
		// Check to ensure we have selected only a contiguous block of
		// cells
		int numcols = table.getColumnCount();
		int numrows = table.getRowCount();
		
		for (int i=0;i<numrows;i++) {
			for (int j=0;j<numcols;j++) {
				sbf.append(table.getValueAt(i,j));
				if (j<numcols-1) sbf.append("\t");
				}
			sbf.append("\n");
			}
		stsel  = new StringSelection(sbf.toString());
		system = Toolkit.getDefaultToolkit().getSystemClipboard();
		system.setContents(stsel,stsel);
  	}
  
  public void copySelectedRows() {
		Clipboard system;
		StringSelection stsel;
		
		StringBuffer sbf = new StringBuffer();
		// Check to ensure we have selected only a contiguous block of
		// cells
		int numcols = table.getColumnCount();
		int numrows = table.getSelectedRowCount();
		int[] rowsselected = table.getSelectedRows();
		
		for (int i=0;i<numrows;i++) {
			for (int j=0;j<numcols;j++) {
				sbf.append(table.getValueAt(rowsselected[i],j));
				if (j<numcols-1) sbf.append("\t");
				}
			sbf.append("\n");
			}
		stsel  = new StringSelection(sbf.toString());
		system = Toolkit.getDefaultToolkit().getSystemClipboard();
		system.setContents(stsel,stsel);
  	}

  // ExprAccessible interface
	public Expr getExpr() {return table.getExpr();}
	public void setExpr(Expr e) {table.setExpr(e);}

	public Expr getPart(int i) {return table.getPart(i);}
	public Expr getPart(int[] ia) {return table.getPart(ia);}
	
	public void setPart(int i, Expr e) {table.setPart(i,e);}
	public void setPart(int[] ia, Expr e) {table.setPart(ia,e);}
	
  // ItemAccessible interface
  
  public Object getItems() {return table.getItems();}
  public void setItems(Object objs) {table.setItems(objs);}
  
  public Object getItem(int index) {return table.getItem(index);}
  public void setItem(int index, Object obj) {table.setItem(index,obj);}
  
  private class SystemPropertiesJTable extends ExprAccessibleJTable {
    private static final long serialVersionUID = -1287787975457788748L;
    public SystemPropertiesJTable(SortTableModel dm, TableColumnModel cm) {
      super(dm, cm);
      }
      
     public String getToolTipText(MouseEvent event) {
      String tip = null;
      Point p = event.getPoint();
      
      // Locate the renderer under the event location
      int hitColumnIndex = columnAtPoint(p);
      int hitRowIndex = rowAtPoint(p);
  
      if ((hitColumnIndex != -1) && (hitRowIndex != -1)) {
        tip = (String)getModel().getValueAt(hitRowIndex, hitColumnIndex);
        // we linebreak at a certain length for really long values
        if (tip != null) {
        	tip = multilineHTMLTooltip(tip, TOOLTIP_LINEBREAK);
          }
        }
      // No tip from the renderer get our own tip
      if (tip == null)
        tip = getToolTipText();
      return tip;
      }
        
    }
    
   /**
   * TableModel for the SystemProperties properties table.
   * Uses a TreeMap as it's data structure.
   *
   * This model is immutable.
   */
  private class SystemPropertiesTableModel extends ItemTableModel {
    private static final long serialVersionUID = -1287987972452788928L;
    private TreeMap treeMap;
  
    // keys and values cache from the map.
    private Object[] keys;
    private Object[] values;

    public SystemPropertiesTableModel(TreeMap map) {
      this.treeMap = map;

      // We need to convert the TreeMap to a
      // couple of arrays to satisfy the 
      // indices arguments of the getValueAt method.
      
      keys = new Object[getRowCount()];
      values = new Object[getRowCount()];

      keys = treeMap.keySet().toArray();
      values = treeMap.values().toArray();
      }
  
    public Object getItems() {
      int count = keys.length;
      Object[][] rowObjects = new Object[count][2];
      for (int i = 0; i < count; ++i) {
        rowObjects[i][0] = keys[i];
        rowObjects[i][1] = values[i];
        }
      return rowObjects;
      }
    
    public Object getItem(int index) {
      if (index < 0 || index >= keys.length) return null;
      return new Object[]{keys[index], values[index]};
      }
    
    public boolean isCellEditable(int row, int column) {
      return false;
      }
    
    //
    // AbstractTableModel methods
    //
    public int getRowCount() {
      if (treeMap != null) return treeMap.size();
      else return 0;
      }
    public int getColumnCount() {return 2;}

    public Object getValueAt(int row, int column) {
      if (column == 0) {
        // Keys
        return keys[row];
        } 
      else if (column == 1) {
        return values[row];
        }
      return null;
      }
      
    }
    
	class PopupListener extends MouseAdapter {
		public void mousePressed(MouseEvent e) {
			maybeShowPopup(e);
			}
		public void mouseReleased(MouseEvent e) {
			maybeShowPopup(e);
			}
		private void maybeShowPopup(MouseEvent e) {
			if (e.isPopupTrigger()) {
				popup.show(e.getComponent(), e.getX(), e.getY());
				}
			}
		}
		
  }

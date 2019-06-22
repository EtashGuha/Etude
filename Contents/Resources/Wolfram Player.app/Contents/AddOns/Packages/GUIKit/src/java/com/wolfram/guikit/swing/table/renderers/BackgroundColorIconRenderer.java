package com.wolfram.guikit.swing.table.renderers;

import java.awt.Color;
import java.awt.Component;

import javax.swing.BorderFactory;
import javax.swing.Icon;
import javax.swing.JLabel;
import javax.swing.JTable;
import javax.swing.border.Border;
import javax.swing.table.TableCellRenderer;

import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.jlink.Expr;

public class BackgroundColorIconRenderer implements TableCellRenderer {
 
  private Border unselectedBorder = null;
  private Border selectedBorder = null;
  private boolean useBorder = false;
  private Color color = Color.WHITE;
  private JLabel label = new JLabel();
  
  public BackgroundColorIconRenderer() {
    label.setOpaque(true); //MUST do this for background to show up.
    label.setHorizontalAlignment(JLabel.CENTER);
    }
  
  public boolean getUseBorder() {return useBorder;}
  public void setUseBorder(boolean val) {
    useBorder = val;
    }
  
  public Color getColor() {return color;}
  public void setColor(Color c) {
    color = c;
    }
    
  public Expr getColorExpr() {
    Expr e = null;
    try {
      e = (Expr)MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
      Color.class, getColor(), Expr.class);
      }
    catch (MathematicaBSFException ex) {}
    return e;
    }
  public void setColorExpr(Expr c) {
    Color col = null;
    if (c != null) {
      try {
        col = (Color)MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
          Expr.class, c, Color.class);
        }
      catch (MathematicaBSFException ex) {}
      }
    setColor(col);
    }
    
  public Component getTableCellRendererComponent(JTable table, Object value,
      boolean isSelected, boolean hasFocus, int row, int column) {
    Color newColor = color;
    // If the cell itself has a color object use this color else use
    // the renderer's instance color
    if (value != null && value instanceof Color) newColor = (Color)value;
 
    label.setBackground(newColor);
    label.setIcon((value instanceof Icon) ? (Icon)value : null); 
    
    if (useBorder) {
      if (isSelected) {
        if (selectedBorder == null) {
          selectedBorder = BorderFactory.createMatteBorder(2,5,2,5,
            table.getSelectionBackground());
          }
        label.setBorder(selectedBorder);
        } 
      else {
        if (unselectedBorder == null) {
          unselectedBorder = BorderFactory.createMatteBorder(2,5,2,5,
            table.getBackground());
          }
        label.setBorder(unselectedBorder);
        }
      }
    return label;
    }
                            
}

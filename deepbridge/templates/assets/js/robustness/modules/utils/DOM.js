// DOM utility functions
const DOM = {
    create: function(tagName, attributes, children) {
        const element = document.createElement(tagName);
        
        if (attributes) {
            for (const [key, value] of Object.entries(attributes)) {
                if (key === 'className') {
                    element.className = value;
                } else if (key === 'dataset') {
                    for (const [dataKey, dataValue] of Object.entries(value)) {
                        element.dataset[dataKey] = dataValue;
                    }
                } else {
                    element.setAttribute(key, value);
                }
            }
        }
        
        if (children) {
            if (Array.isArray(children)) {
                children.forEach(child => {
                    if (child) {
                        if (typeof child === 'string') {
                            element.appendChild(document.createTextNode(child));
                        } else {
                            element.appendChild(child);
                        }
                    }
                });
            } else if (typeof children === 'string') {
                element.textContent = children;
            } else {
                element.appendChild(children);
            }
        }
        
        return element;
    }
};

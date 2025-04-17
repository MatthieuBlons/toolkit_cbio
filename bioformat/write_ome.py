import pyvips
import ome_types
import yaml


def save_ometif(
    image_pyvips,
    dst,
    pixel_size_x,
    pixel_size_y,
    magnification,
    channel_xml,
    channel_names,
    channel_colors,
    CONFIG_PATH=None,
    PYVIPS2OME_FORMAT="./PYVIPS2OME_FORMAT.yaml",
):
    # collect image dimension needed for OME-XML before separating image planes
    width = image_pyvips.width
    height = image_pyvips.height
    bands = image_pyvips.bands
    format_str = image_pyvips.format

    with open(PYVIPS2OME_FORMAT, "r") as f:
        formats = yaml.safe_load(f)

    ome_format = formats[format_str]

    # split to separate image planes and stack vertically for OME-TIFF
    image_pyvips = pyvips.Image.arrayjoin(image_pyvips.bandsplit(), across=1)
    # Set tiff tags necessary for OME-TIFF
    image_pyvips = image_pyvips.copy()

    if CONFIG_PATH:
        DEFAUT_CONFIG = ome_types.from_xml(CONFIG_PATH)
        ome_xml_metadata = adapt_ome_metadata(
            xml_config=DEFAUT_CONFIG,
            width=width,
            height=height,
            bands=bands,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            channel_names=channel_names,
            channel_colors=channel_colors,
            magnification=magnification,
            ome_format=ome_format,
        )
    else:
        # build minimal OME metadata.
        ome_xml_metadata = simple_ome_metadata(
            width,
            height,
            bands,
            pixel_size_x,
            pixel_size_y,
            magnification,
            channel_xml,
            format=ome_format,
        )

    image_pyvips.set_type(
        pyvips.GValue.gstr_type, "image-description", ome_xml_metadata
    )
    image_pyvips.set_type(pyvips.GValue.gint_type, "page-height", height)

    image_pyvips.write_to_file(
        dst,
        compression="deflate",
        predictor="none",
        tile=True,
        tile_width=512,
        tile_height=512,
        pyramid=True,
        subifd=True,
        bigtiff=True,
    )


def adapt_ome_metadata(
    xml_config,
    width,
    height,
    bands,
    pixel_size_x,
    pixel_size_y,
    channel_names,
    channel_colors,
    magnification,
    ome_format,
):
    xml_config.images[0].pixels.size_c = bands
    xml_config.images[0].pixels.type = ome_format
    xml_config.images[0].pixels.size_x = width
    xml_config.images[0].pixels.size_y = height
    xml_config.images[0].pixels.physical_size_x = pixel_size_x
    xml_config.images[0].pixels.physical_size_y = pixel_size_y
    planes = [
        ome_types.model.Plane(the_z=0, the_t=0, the_c=idx_c) for idx_c in range(bands)
    ]
    channels = [
        ome_types.model.Channel(
            id=f"Channel:{idx_c}",
            name=channel_names[idx_c],
            color=channel_colors[idx_c],
            samples_per_pixel=1,
            light_path={},
        )
        for idx_c in range(bands)
    ]
    xml_config.images[0].pixels.planes = planes
    xml_config.images[0].pixels.channels = channels
    xml_config.instruments[0].objectives[0].nominal_magnification = magnification
    return xml_config.to_xml()


def simple_ome_metadata(
    width,
    height,
    bands,
    pixel_size_x,
    pixel_size_y,
    magnification,
    channels_xml,
    format="float",
):
    ome_xml_metadata = f"""<?xml version="1.0" encoding="UTF-8"?>
                        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
                            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                            xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
                              <Instrument ID="Instrument:0">
                                <Objective ID="Objective:0" NominalMagnification="{magnification}"/>
                                </Instrument>
                            <Image ID="Image:0">
                                <!-- Minimum required fields about image dimensions -->
                                <Pixels DimensionOrder="XYCZT"
                                        ID="Pixels:0"
                                        SizeC="{bands}"
                                        SizeT="1"
                                        SizeX="{width}"
                                        SizeY="{height}"
                                        SizeZ="1"
                                        Type={format}                    
                                        PhysicalSizeX="{pixel_size_x}"
                                        PhysicalSizeY="{pixel_size_y}"
                                        PhysicalSizeXUnit="µm"
                                        PhysicalSizeYUnit="µm">
                                        {channels_xml} 
                                </Pixels>
                            </Image>
                        </OME>"""
    return ome_xml_metadata
